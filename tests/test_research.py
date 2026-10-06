import pandas as pd
import pytest

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
