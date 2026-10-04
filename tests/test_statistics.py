import numpy as np
import pytest
from src.statistics import wilson_interval, holm_adjust, event_bootstrap

def test_intervals_and_multiplicity():
    low,high=wilson_interval(5,10)
    assert low<.5<high
    assert low==pytest.approx(1-high)
    assert np.allclose(holm_adjust([.01,.03,.8]),[.03,.06,.8])
    assert event_bootstrap([2,2,2],['a','b','c'])==pytest.approx((2,2))
