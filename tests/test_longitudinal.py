import numpy as np
import pandas as pd
import pytest

from src.data_loader import TABLES


def test_elo_development_predictions_share_fights_and_reserve_holdout():
    rows = pd.read_csv(TABLES / "deep_elo_predictions.csv", parse_dates=["event_date"])
    assert rows.event_date.dt.year.between(2018, 2022).all()
    assert rows.probability.between(0, 1).all()
    assert not rows.duplicated(["fight_id", "model"]).any()
    assert rows.groupby("fight_id").model.nunique().eq(6).all()
    wide = rows.pivot(
        index=["fight_id", "a_won"], columns="model", values="probability"
    ).reset_index()

    def loss(p):
        return -(wide.a_won * np.log(p) + (1 - wide.a_won) * np.log(1 - p))

    delta = loss(wide["Historical logistic"]) - loss(wide["History plus Elo"])
    gains = pd.read_csv(TABLES / "deep_elo_gain.csv")
    assert gains.log_loss_reduction.iloc[0] == pytest.approx(delta.mean())


def test_layoff_calipers_and_reported_denominators_reconcile():
    pairs = pd.read_csv(TABLES / "deep_layoff_pairs.csv")
    assert pairs.fight_id.is_unique
    assert pairs.rest_gap.ge(90).all()
    assert (pairs.r_days_since - pairs.b_days_since).abs().to_numpy() == pytest.approx(
        pairs.rest_gap.to_numpy()
    )
    matched = pairs[pairs.age_gap.le(2) & pairs.elo_gap.le(50)]
    reported = pd.read_csv(TABLES / "deep_layoff_matched.csv").iloc[-1]
    assert len(matched) == reported.n
    assert matched.longer_rest_won.mean() == pytest.approx(reported.win_rate)
