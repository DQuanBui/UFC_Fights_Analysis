import numpy as np
import pandas as pd
import pytest

from src.ratings import expected_score, pre_fight_state


def sample():
    return pd.DataFrame(
        {
            "fight_id": ["first", "same_a", "same_b", "later", "other"],
            "event_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-02-01", "2020-03-01", "2019-01-01"]
            ),
            "r_id": ["a"] * 5,
            "b_id": ["b", "b", "c", "b", "c"],
            "winner_id": ["a", "b", "a", "b", "c"],
            "decisive": [True] * 5,
            "result_status": ["win"] * 5,
            "is_ufc": [True, True, True, True, False],
        }
    )


def test_ratings_freeze_same_day_and_exclude_other_promotions():
    frame = pre_fight_state(sample()).set_index("fight_id")
    assert "other" not in frame.index
    assert frame.loc["first", "r_elo"] == 1500
    assert np.isnan(frame.loc["first", "r_days_since"])
    assert frame.loc["same_a", "r_elo"] == frame.loc["same_b", "r_elo"] == 1516
    assert frame.loc["same_b", "r_days_since"] == 31
    assert frame.loc["same_b", "r_prior_bouts"] == 1
    assert frame.loc["later", "r_prior_bouts"] == 3
    pd.testing.assert_frame_equal(
        pre_fight_state(sample()),
        pre_fight_state(sample().sample(frac=1, random_state=5)),
    )


def test_future_and_current_outcomes_cannot_change_pre_fight_state():
    original = sample()
    changed = original.copy()
    changed.loc[changed.event_date.ge("2020-02-01"), "winner_id"] = "a"
    before = pre_fight_state(original)
    after = pre_fight_state(changed)
    pd.testing.assert_frame_equal(
        before[before.event_date.le("2020-02-01")],
        after[after.event_date.le("2020-02-01")],
    )
    assert not before.equals(after)


def test_draw_updates_and_no_contest_does_not():
    fights = sample().query("fight_id != 'same_b'").copy()
    fights.loc[fights.fight_id.eq("same_a"), ["decisive", "result_status"]] = [
        False,
        "draw",
    ]
    draw = pre_fight_state(fights).set_index("fight_id")
    assert 1500 < draw.loc["later", "r_elo"] < 1516
    assert draw.loc["later", "r_elo"] + draw.loc["later", "b_elo"] == pytest.approx(
        3000
    )
    fights.loc[fights.fight_id.eq("same_a"), "result_status"] = "no_contest"
    nc = pre_fight_state(fights).set_index("fight_id")
    assert nc.loc["later", "r_elo"] == 1516
    assert nc.loc["later", "r_days_since"] == 29


def test_expected_score_is_symmetric_and_parameters_are_validated():
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1)
    with pytest.raises(ValueError, match="positive"):
        pre_fight_state(sample(), k=0)
