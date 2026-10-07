import json

import nbformat
import pandas as pd
import pytest

from src.data_loader import ROOT, TABLES
from src.research import product_decomposition, rate_decomposition
from src.research_models import expanding_splits


def test_ablation_folds_respect_dates_and_reserve_test_period():
    frame = pd.DataFrame(
        {
            "fight_year": [2017, 2018, 2018, 2019, 2023],
            "event_date": pd.to_datetime(
                ["2017-12-01", "2018-02-01", "2018-02-01", "2019-05-01", "2023-01-01"]
            ),
        }
    )
    splits = list(expanding_splits(frame, years=[2018, 2019]))
    assert [len(s[2]) for s in splits] == [2, 1]
    assert all(
        train.event_date.max() < validation.event_date.min()
        for _, train, validation in splits
    )
    with pytest.raises(ValueError, match="2023"):
        list(expanding_splits(frame, years=[2023]))


def test_within_and_mix_contributions_reconcile():
    frame = pd.DataFrame(
        {
            "weight_class": ["X"] * 4 + ["Y"] * 4,
            "period": ["Before", "Before", "After", "After"] * 2,
            "is_decision": [0, 0, 1, 1, 1, 0, 1, 0],
        }
    )
    result = rate_decomposition(frame, minimum=1)
    before = frame.loc[frame.period.eq("Before"), "is_decision"].mean()
    after = frame.loc[frame.period.eq("After"), "is_decision"].mean()
    assert result.total_contribution.sum() == pytest.approx(after - before)
    assert result.mix_contribution.sum() == pytest.approx(0)


def test_growth_contributions_reconcile_and_reverse():
    first, second = product_decomposition(10, 15, 8, 12)
    assert first + second == pytest.approx(15 * 12 - 10 * 8)
    assert product_decomposition(15, 10, 12, 8) == pytest.approx((-first, -second))
    assert product_decomposition(10, 10, 8, 12) == pytest.approx((0, 40))


def test_saved_career_followup_has_equal_observation_window():
    careers = pd.read_csv(
        TABLES / "deep_career_followup.csv", parse_dates=["debut_date"]
    )
    fights = pd.read_csv(
        ROOT / "data" / "processed" / "fights.csv",
        usecols=["event_date", "is_ufc", "r_id", "b_id"],
        parse_dates=["event_date"],
    )
    fights = fights[fights.is_ufc]
    endpoint = fights.event_date.max()
    assert careers.debut_date.le(endpoint - pd.Timedelta(days=730)).all()
    for row in careers.sample(20, random_state=42).itertuples():
        actual = fights[
            (fights.r_id.eq(row.fighter_id) | fights.b_id.eq(row.fighter_id))
            & fights.event_date.between(
                row.debut_date, row.debut_date + pd.Timedelta(days=730)
            )
        ]
        assert len(actual) == row.fights_within_730_days
        assert row.reached_three_fights == (len(actual) >= 3)


def test_bonus_standardization_keeps_common_denominators():
    table = pd.read_csv(TABLES / "deep_bonus_division_standardization.csv")
    assert table.n.ge(50).all()
    assert table.successes.le(table.n).all()
    weights = table.pivot(
        index="weight_class", columns="outcome", values="pooled_weight"
    )
    assert weights.Finish.to_numpy() == pytest.approx(weights.Decision.to_numpy())
    assert weights.Finish.sum() == pytest.approx(1)
    counts = table.groupby("weight_class").n.sum()
    assert weights.Finish.to_numpy() == pytest.approx(
        (counts / counts.sum()).to_numpy()
    )


def test_paired_model_gain_matches_saved_predictions():
    from src.modeling import model_frame
    from src.pipeline import prepare

    frame = model_frame(prepare(False)["fights"]).set_index("fight_id")
    selected = json.loads((TABLES / "model_metadata.json").read_text())[
        "selected_model"
    ]
    predictions = pd.read_csv(TABLES / "test_predictions.csv")
    predictions = predictions[predictions.model.eq(selected)]
    correct = predictions.p_a_wins.ge(0.5).eq(predictions.a_won)
    red_correct = (
        frame.loc[predictions.fight_id, "source_red_is_a"].to_numpy()
        == predictions.a_won.to_numpy()
    )
    gains = pd.read_csv(TABLES / "deep_paired_model_gain.csv").set_index("comparator")
    assert gains.loc["Source red-corner heuristic", "accuracy_gain"] == pytest.approx(
        correct.mean() - red_correct.mean()
    )


def test_notebook_rebuild_preserves_all_research_chapters(tmp_path, monkeypatch):
    from src import notebooks
    from src.research_notebooks import EXTRA_CHAPTERS

    monkeypatch.setattr(notebooks, "ROOT", tmp_path)
    paths = notebooks.build_notebooks()
    assert len(paths) == 8
    for path in paths:
        book = nbformat.read(path, 4)
        research = [
            c.source
            for c in book.cells
            if "research-extension" in c.metadata.get("tags", [])
        ]
        assert any(f"run_chapter('{path.name[:2]}')" in text for text in research)
        assert any("research_chart(" in text for text in research)
        for extra in EXTRA_CHAPTERS.get(path.name[:2], []):
            assert any(f"run_chapter('{extra}')" in text for text in research)
