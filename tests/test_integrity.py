import json

import numpy as np
import pandas as pd
import pytest

from src.analysis import annual_summary, composition_adjustment
from src.data_loader import ROOT, load_raw, verify_raw
from src.modeling import FEATURES, model_frame
from src.pipeline import prepare


@pytest.fixture(scope="module")
def data():
    return prepare(False)


def test_original_snapshot_and_relational_grains(data):
    assert verify_raw() == 7
    raw = load_raw()
    assert set(data["fights"].fight_id) == set(raw["fight"].fight_id)
    assert data["fights"].fight_id.is_unique
    assert not data["appearances"].duplicated(["fight_id", "fighter_id"]).any()
    assert len(data["appearances"]) == 2 * len(data["fights"])
    assert not data["rounds"].duplicated(["fight_id", "fighter_id", "round_no"]).any()


def test_denominators_and_missing_not_zero(data):
    f = data["fights"].query("in_scope")
    a = data["appearances"]
    assert len(f) == 8820 and f.event_id.nunique() == 783
    assert f.outcome_group.value_counts(normalize=True).sum() == pytest.approx(1)
    assert a.loc[~a.round_stats_complete, "sig_landed"].isna().all()
    assert a.loc[~a.round_stats_complete, "td_success"].isna().all()
    assert a.loc[a.sig_atmp.eq(0), "sig_accuracy"].isna().all()
    assert f.fight_duration_seconds.gt(0).all()
    assert int((~f.round_stats_complete).sum()) == 21


def test_complete_master_totals_match(data):
    raw = load_raw()
    master = raw["master"].set_index("fight_id")
    f = data["fights"]
    a = data["appearances"]
    red = a.merge(f[["fight_id", "r_id"]], on="fight_id", validate="many_to_one")
    red = red[red.fighter_id.eq(red.r_id) & red.round_stats_complete].set_index(
        "fight_id"
    )
    assert np.allclose(red.sig_landed, master.loc[red.index, "r_total_sig_landed"])
    assert np.allclose(red.td_success, master.loc[red.index, "r_total_td_success"])


def test_direct_historical_count_and_current_feature_exclusion(data):
    f = data["fights"]
    a = data["appearances"]
    row = f[f.in_scope].sort_values("event_date").iloc[-1]
    prior = a[a.is_ufc & a.fighter_id.eq(row.r_id) & a.event_date.lt(row.event_date)]
    assert row.r_prior_ufc_fights == len(prior)
    assert row.r_prior_wins == prior.won.sum()
    modified = f.copy()
    modified["method"] = "changed"
    modified["is_ko"] = False
    original = model_frame(f)
    changed = model_frame(modified)
    pd.testing.assert_frame_equal(original[FEATURES], changed[FEATURES])
    assert original.groupby("event_date").split.nunique().max() == 1
    assert not original.fight_id.duplicated().any()


def test_insufficient_composition_support_returns_empty(data):
    assert composition_adjustment(data["fights"].query("fight_year == 2025")).empty


def test_full_selected_year_is_not_marked_partial(data):
    f = data["fights"].query("in_scope and fight_year == 2025")
    a = data["appearances"].query("in_scope and fight_year == 2025")
    assert not annual_summary(f, a).partial_year.any()


def test_saved_prediction_metrics_reconcile():
    table = ROOT / "outputs" / "tables"
    selected = json.loads((table / "model_metadata.json").read_text())["selected_model"]
    predicted = pd.read_csv(table / "test_predictions.csv")
    predicted = predicted[predicted.model.eq(selected)]
    scores = pd.read_csv(table / "model_metrics.csv")
    scores = scores[scores.model.eq(selected) & scores.split.eq("test")].iloc[0]
    assert (predicted.p_a_wins.ge(0.5) == predicted.a_won).mean() == pytest.approx(
        scores.accuracy
    )
    assert predicted.p_a_wins.between(0, 1).all()
