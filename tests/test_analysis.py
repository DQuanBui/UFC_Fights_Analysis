import pandas as pd
import pytest

from src.analysis import duration_curve, round_hazard


def test_hazard_denominator_includes_only_reached_rounds():
    fights = pd.DataFrame(
        {
            "standard_format": [True] * 3,
            "scheduled_rounds": [3] * 3,
            "finish_round": [1, 2, 3],
            "is_finish": [True, True, False],
            "fight_duration_seconds": [90, 400, 900],
        }
    )
    hazard = round_hazard(fights)
    assert hazard.at_risk.tolist() == [3, 2, 1]
    assert hazard.finish_risk.tolist() == pytest.approx([1 / 3, 1 / 2, 0])
    assert duration_curve(fights).still_fighting.iloc[-1] == 0
