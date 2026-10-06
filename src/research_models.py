"""Development-period feature ablations and fixed-holdout diagnostics."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .modeling import FEATURES, NUMERIC, evaluate, model_frame
from .statistics import event_bootstrap, wilson_interval


def expanding_splits(frame, years=(2018, 2019, 2020, 2021, 2022)):
    """Reserve 2023 onward; every evaluation year follows all training dates."""
    if any(year >= 2023 for year in years):
        raise ValueError("Feature ablations must stay before the existing 2023 holdout")
    for year in years:
        train = frame[frame.fight_year.lt(year)]
        validation = frame[frame.fight_year.eq(year)]
        if train.empty or validation.empty:
            continue
        if train.event_date.max() >= validation.event_date.min():
            raise ValueError("Chronological split overlaps")
        yield year, train, validation


def feature_ablation(fights):
    frame = model_frame(fights)
    groups = {
        "Age and division": ["age_diff"],
        "Age, records and division": [
            c for c in NUMERIC if not any(x in c for x in ["sig_", "td_"])
        ],
        "Full historical features": NUMERIC,
    }
    results = []
    for year, train, validation in expanding_splits(frame):
        baseline = DummyClassifier(strategy="prior").fit(train[FEATURES], train.a_won)
        probability = baseline.predict_proba(validation[FEATURES])[:, 1]
        results.append(
            dict(
                year=year,
                features="Majority baseline",
                train_fights=len(train),
                n=len(validation),
                **evaluate(validation.a_won, probability),
            )
        )
        for name, columns in groups.items():
            numeric = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scale", StandardScaler()),
                ]
            )
            transform = ColumnTransformer(
                [
                    ("numeric", numeric, columns),
                    (
                        "division",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ["weight_class"],
                    ),
                ]
            )
            model = Pipeline(
                [
                    ("transform", transform),
                    (
                        "model",
                        LogisticRegression(C=0.3, max_iter=1500, random_state=42),
                    ),
                ]
            )
            model.fit(train[FEATURES], train.a_won)
            probability = model.predict_proba(validation[FEATURES])[:, 1]
            results.append(
                dict(
                    year=year,
                    features=name,
                    train_fights=len(train),
                    n=len(validation),
                    **evaluate(validation.a_won, probability),
                )
            )
    return pd.DataFrame(results)


def holdout_diagnostics(fights, predictions):
    frame = model_frame(fights).set_index("fight_id")
    predictions = predictions.copy()
    predictions["correct"] = predictions.p_a_wins.ge(0.5).eq(predictions.a_won)
    predictions["confidence"] = np.maximum(
        predictions.p_a_wins, 1 - predictions.p_a_wins
    )
    predictions["confidence_band"] = pd.cut(
        predictions.confidence, [0.5, 0.55, 0.60, 0.65, 0.70, 0.80, 1.01], right=False
    )
    confidence = []
    for band, rows in predictions.groupby("confidence_band", observed=True):
        low, high = wilson_interval(rows.correct.sum(), len(rows))
        confidence.append(
            dict(
                band=str(band),
                n=len(rows),
                mean_confidence=rows.confidence.mean(),
                accuracy=rows.correct.mean(),
                low=low,
                high=high,
            )
        )
    year = []
    for date, rows in predictions.groupby(
        pd.to_datetime(predictions.event_date).dt.year
    ):
        year.append(
            dict(
                year=date,
                n=len(rows),
                partial_year=date == 2026,
                **evaluate(rows.a_won, rows.p_a_wins),
            )
        )
    differences = []
    for name, baseline in [
        ("Majority baseline", np.zeros(len(predictions), dtype=int)),
        (
            "Source red-corner heuristic",
            frame.loc[predictions.fight_id, "source_red_is_a"].astype(int).to_numpy(),
        ),
    ]:
        delta = predictions.correct.astype(int).to_numpy() - (
            baseline == predictions.a_won.to_numpy()
        ).astype(int)
        low, high = event_bootstrap(delta, predictions.event_id.to_numpy())
        differences.append(
            dict(
                comparator=name,
                n=len(predictions),
                accuracy_gain=delta.mean(),
                event_bootstrap_low=low,
                event_bootstrap_high=high,
            )
        )
    return pd.DataFrame(confidence), pd.DataFrame(year), pd.DataFrame(differences)
