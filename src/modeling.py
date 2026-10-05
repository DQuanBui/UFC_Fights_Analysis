"""Pre-fight prediction with fixed calendar splits and training-only preprocessing."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .analysis import save_tables
from .data_loader import ROOT
from .statistics import event_bootstrap

BASE_FEATURES = [
    "age",
    "prior_ufc_fights",
    "prior_wins",
    "prior_losses",
    "prior_win_rate",
    "prior_ko_rate",
    "prior_submission_rate",
    "prior_win_streak",
    "prior_title_fights",
    "prior_sig_per_minute",
    "prior_sig_accuracy",
    "prior_td_per_15",
    "prior_td_accuracy",
]
NUMERIC = [f"{col}_diff" for col in BASE_FEATURES]
CATEGORICAL = ["weight_class"]
FEATURES = NUMERIC + CATEGORICAL


def model_frame(fights):
    """Assign A/B independently of outcome, and expose only an explicit feature allowlist."""
    rows = (
        fights[fights.in_scope & fights.decisive]
        .sort_values(["event_date", "fight_id"])
        .copy()
    )
    a_red = rows.fight_id.map(
        lambda value: int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 2 == 0
    )
    sign = np.where(a_red, 1, -1)
    frame = rows[
        ["fight_id", "event_id", "event_date", "fight_year", "weight_class"]
    ].copy()
    frame["a_id"] = rows.r_id.where(a_red, rows.b_id)
    frame["b_id"] = rows.b_id.where(a_red, rows.r_id)
    frame["a_name"] = rows.r_fighter_name.where(a_red, rows.b_fighter_name)
    frame["b_name"] = rows.b_fighter_name.where(a_red, rows.r_fighter_name)
    frame["a_won"] = rows.winner_id.eq(frame.a_id).astype(int)
    frame["source_red_is_a"] = a_red
    frame["has_debutant"] = rows.r_prior_ufc_fights.eq(0) | rows.b_prior_ufc_fights.eq(
        0
    )
    for feature in BASE_FEATURES:
        frame[feature + "_diff"] = (rows["r_" + feature] - rows["b_" + feature]) * sign
    frame["split"] = np.select(
        [frame.fight_year.le(2019), frame.fight_year.le(2022)],
        ["train", "validation"],
        default="test",
    )
    return frame.reset_index(drop=True)


def build_pipeline(estimator):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prep = ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
    )
    return Pipeline([("preprocess", prep), ("model", estimator)])


def evaluate(y, probabilities):
    predicted = np.asarray(probabilities) >= 0.5
    return dict(
        accuracy=accuracy_score(y, predicted),
        precision=precision_score(y, predicted, zero_division=0),
        recall=recall_score(y, predicted, zero_division=0),
        f1=f1_score(y, predicted, zero_division=0),
        roc_auc=roc_auc_score(y, probabilities),
        brier_score=brier_score_loss(y, probabilities),
        log_loss=log_loss(y, probabilities, labels=[0, 1]),
    )


def train_models(fights):
    frame = model_frame(fights)
    train = frame[frame.split.eq("train")]
    validation = frame[frame.split.eq("validation")]
    test = frame[frame.split.eq("test")]
    assert train.event_date.max() < validation.event_date.min() < test.event_date.min()
    candidates = {
        "Majority baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(
            C=0.3, max_iter=1500, random_state=42
        ),
        "Decision tree": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=60, random_state=42
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=35,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=7,
            learning_rate=0.05,
            l2_regularization=5,
            random_state=42,
        ),
    }
    scores = []
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(train[FEATURES], train.a_won)
        p = pipe.predict_proba(validation[FEATURES])[:, 1]
        scores.append(
            dict(
                model=name,
                split="validation",
                n=len(validation),
                **evaluate(validation.a_won, p),
            )
        )
    validation_scores = pd.DataFrame(scores)
    selected = validation_scores.sort_values(["log_loss", "model"]).iloc[0]["model"]
    development = frame[~frame.split.eq("test")]
    predictions = []
    curves = []
    saved = {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(development[FEATURES], development.a_won)
        p = pipe.predict_proba(test[FEATURES])[:, 1]
        low, high = event_bootstrap(
            (p >= 0.5) == test.a_won.to_numpy(), test.event_id.to_numpy()
        )
        scores.append(
            dict(
                model=name,
                split="test",
                n=len(test),
                accuracy_ci_low=low,
                accuracy_ci_high=high,
                **evaluate(test.a_won, p),
            )
        )
        prediction = test[
            [
                "fight_id",
                "event_id",
                "event_date",
                "weight_class",
                "a_id",
                "b_id",
                "a_name",
                "b_name",
                "a_won",
                "has_debutant",
            ]
        ].copy()
        prediction["model"] = name
        prediction["p_a_wins"] = p
        predictions.append(prediction)
        fpr, tpr, threshold = roc_curve(test.a_won, p)
        curves.append(
            pd.DataFrame(
                {
                    "model": name,
                    "false_positive_rate": fpr,
                    "true_positive_rate": tpr,
                    "threshold": threshold,
                }
            )
        )
        saved[name] = pipe
    # Source corner ordering can encode matchmaking strength; show it as a hard-label comparator.
    for split, subset in [("validation", validation), ("test", test)]:
        predicted = subset.source_red_is_a.astype(int)
        scores.append(
            dict(
                model="Source red-corner heuristic",
                split=split,
                n=len(subset),
                accuracy=accuracy_score(subset.a_won, predicted),
                precision=precision_score(subset.a_won, predicted),
                recall=recall_score(subset.a_won, predicted),
                f1=f1_score(subset.a_won, predicted),
            )
        )
    predictions = pd.concat(predictions, ignore_index=True)
    chosen = predictions[predictions.model.eq(selected)]
    matrix = confusion_matrix(chosen.a_won, chosen.p_a_wins.ge(0.5), labels=[0, 1])
    confusion = pd.DataFrame(
        {"actual": [0, 0, 1, 1], "predicted": [0, 1, 0, 1], "count": matrix.ravel()}
    )
    fraction, mean = calibration_curve(
        chosen.a_won, chosen.p_a_wins, n_bins=8, strategy="quantile"
    )
    calibration = pd.DataFrame(
        {"mean_prediction": mean, "observed_a_win_rate": fraction}
    )
    logistic = saved["Logistic regression"]
    coefficients = pd.DataFrame(
        {
            "feature": logistic["preprocess"].get_feature_names_out(),
            "coefficient": logistic["model"].coef_[0],
        }
    ).sort_values("coefficient")
    subgroup = []
    for field in ["weight_class", "has_debutant"]:
        for label, rows in chosen.groupby(field):
            if len(rows) >= 30 and rows.a_won.nunique() == 2:
                subgroup.append(
                    dict(
                        group=field,
                        value=label,
                        n=len(rows),
                        **evaluate(rows.a_won, rows.p_a_wins),
                    )
                )
    tables = {
        "model_metrics": pd.DataFrame(scores),
        "test_predictions": predictions,
        "roc_curves": pd.concat(curves),
        "model_confusion": confusion,
        "model_calibration": calibration,
        "model_coefficients": coefficients,
        "model_subgroups": pd.DataFrame(subgroup),
        "model_splits": frame.groupby("split", sort=False)
        .agg(
            fights=("fight_id", "size"),
            first_date=("event_date", "min"),
            last_date=("event_date", "max"),
            a_win_rate=("a_won", "mean"),
        )
        .reset_index(),
    }
    save_tables(tables)
    model_dir = ROOT / "outputs" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(saved[selected], model_dir / "winner_model.joblib")
    metadata = {
        "selected_model": selected,
        "selection_metric": "lowest 2020–2022 validation log loss",
        "train": "1994–2019",
        "validation": "2020–2022",
        "test": "2023–2026 snapshot end",
        "features": FEATURES,
        "final_fit": "1994–2022",
        "evaluation": "Sequential historical replay: earlier test outcomes may update later histories; model weights stay fixed.",
    }
    (ROOT / "outputs" / "tables" / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return tables
