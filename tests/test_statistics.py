import numpy as np
import pytest

from src.statistics import event_bootstrap, holm_adjust, wilson_interval


def test_intervals_and_multiplicity():
    low, high = wilson_interval(5, 10)
    assert low < 0.5 < high
    assert low == pytest.approx(1 - high)
    assert np.allclose(holm_adjust([0.01, 0.03, 0.8]), [0.03, 0.06, 0.8])
    assert event_bootstrap([2, 2, 2], ["a", "b", "c"]) == pytest.approx((2, 2))
