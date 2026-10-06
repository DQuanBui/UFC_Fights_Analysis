import pytest

from src.research import product_decomposition


def test_growth_contributions_reconcile_and_reverse():
    first, second = product_decomposition(10, 15, 8, 12)
    assert first + second == pytest.approx(15 * 12 - 10 * 8)
    assert product_decomposition(15, 10, 12, 8) == pytest.approx((-first, -second))
    assert product_decomposition(10, 10, 8, 12) == pytest.approx((0, 40))
