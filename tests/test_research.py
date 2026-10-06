import pandas as pd
import pytest

from src.research import product_decomposition, rate_decomposition


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
