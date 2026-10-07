import pandas as pd
import pytest

from src.data_loader import TABLES
from src.generalization import fitting_familiarity, grouped_scores


def test_fitting_membership_stays_fixed_across_test_appearances():
    development = pd.DataFrame({"a_id": ["a"], "b_id": ["b"]})
    test = pd.DataFrame({"a_id": ["a", "a", "c", "c"], "b_id": ["b", "c", "d", "a"]})
    assert fitting_familiarity(development, test).tolist() == [
        "Both in fitting data",
        "One absent from fitting data",
        "Both absent from fitting data",
        "One absent from fitting data",
    ]


def test_small_groups_withhold_metrics():
    frame = pd.DataFrame(
        {
            "slice": ["small"] * 2,
            "event_id": ["a", "b"],
            "correct": [True, False],
            "confidence": [0.6, 0.7],
            "p_a_wins": [0.6, 0.7],
            "a_won": [1, 0],
        }
    )
    row = grouped_scores(frame, "slice", 0.5).iloc[0]
    assert row.n == 2 and not row.reportable
    assert pd.isna(row.accuracy) and pd.isna(row.brier_skill)


def test_familiarity_partition_reconciles_to_predictions():
    predictions = pd.read_csv(TABLES / "deep_generalization_predictions.csv")
    groups = pd.read_csv(TABLES / "deep_generalization_familiarity.csv").set_index(
        "group"
    )
    assert len(predictions) == predictions.fight_id.nunique() == groups.n.sum() == 1851
    for group, part in predictions.groupby("familiarity"):
        assert part.correct.mean() == pytest.approx(groups.loc[group, "accuracy"])
        assert ((part.p_a_wins - part.a_won) ** 2).mean() == pytest.approx(
            groups.loc[group, "brier_score"]
        )
