import numpy as np

from src.modeling import FEATURES, evaluate


def test_feature_allowlist_excludes_outcome_and_snapshot_metrics():
    assert all(
        x == "weight_class" or x == "age_diff" or x.startswith("prior_")
        for x in FEATURES
    )
    assert not any(
        x in FEATURES
        for x in ["winner_id", "method", "r_slpm", "r_sig_landed", "source_red_is_a"]
    )


def test_evaluation_uses_probability_and_binary_target():
    score = evaluate(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.7, 0.8]))
    assert score["accuracy"] == 0.75
    assert score["roc_auc"] == 1
