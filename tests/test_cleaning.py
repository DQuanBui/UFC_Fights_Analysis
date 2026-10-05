import math

import pytest

from src.cleaning import (
    fight_duration,
    is_ufc_event,
    normalize_division,
    parse_clock,
    parse_height,
)


def test_historical_duration_uses_actual_schedule():
    assert fight_duration("3 Rnd (10-5-5)", 2, "2:30") == 750
    assert fight_duration("1 Rnd + 2OT (15-3-3)", 3, "2:00") == 1200
    assert fight_duration("Unlimited Rnd (15)", 6, "1:00") == 4560
    assert fight_duration("No Time Limit", 1, "25:00") == 1500
    assert math.isnan(fight_duration("3 Rnd (5-5-5)", 2, "5:20"))


@pytest.mark.parametrize("bad", ["--", None, "5:99", "-1:00"])
def test_bad_clock_stays_missing(bad):
    assert math.isnan(parse_clock(bad))


def test_scope_and_divisions():
    assert is_ufc_event("Noche UFC: A vs. B")
    assert is_ufc_event("The Ultimate Fighter: Finale")
    assert not is_ufc_event("Road to UFC 1")
    assert not is_ufc_event("DWCS 1.3")
    assert (
        normalize_division("Ultimate Fighter 8 Light Heavyweight Tournament")
        == "Light Heavyweight"
    )
    assert parse_height("5' 10\"") == 70
