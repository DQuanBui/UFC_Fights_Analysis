"""Fixed-model diagnostics for fighter familiarity and pre-fight experience."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .data_loader import TABLES
from .modeling import model_frame
from .research import answer
from .statistics import event_bootstrap, wilson_interval


def fitting_familiarity(development, test):
    """Membership is frozen at model fitting, never updated by test appearances."""
    known = set(development.a_id) | set(development.b_id)
    absent = (~test.a_id.isin(known)).astype(int) + (~test.b_id.isin(known)).astype(int)
    return absent.map(
        {
            0: "Both in fitting data",
            1: "One absent from fitting data",
            2: "Both absent from fitting data",
        }
    )


def grouped_scores(frame, column, prior):
    rows = []
    for group, part in frame.groupby(column, observed=True):
        low, high = wilson_interval(part.correct.sum(), len(part))
        bootstrap_low, bootstrap_high = event_bootstrap(
            part.correct.astype(float), part.event_id
        )
        brier = ((part.p_a_wins - part.a_won) ** 2).mean()
        baseline = ((prior - part.a_won) ** 2).mean()
        rows.append(
            dict(
                group=group,
                n=len(part),
                reportable=len(part) >= 30,
                accuracy=part.correct.mean(),
                low=low,
                high=high,
                event_low=bootstrap_low,
                event_high=bootstrap_high,
                mean_confidence=part.confidence.mean(),
                accuracy_minus_confidence=part.correct.mean() - part.confidence.mean(),
                brier_score=brier,
                reference_brier=baseline,
                brier_skill=1 - brier / baseline,
                log_loss=log_loss(part.a_won, part.p_a_wins, labels=[0, 1]),
            )
        )
    result = pd.DataFrame(rows)
    metrics = result.columns.difference(["group", "n", "reportable"])
    result.loc[~result.reportable, metrics] = np.nan
    return result


def generalization_research(data):
    frame = model_frame(data["fights"])
    development = frame[frame.fight_year.le(2022)]
    metadata = json.loads((TABLES / "model_metadata.json").read_text(encoding="utf-8"))
    saved = pd.read_csv(TABLES / "test_predictions.csv")
    saved = saved[saved.model.eq(metadata["selected_model"])].copy()
    test = frame[frame.split.eq("test")].copy()
    test["familiarity"] = fitting_familiarity(development, test)
    raw = data["fights"].set_index("fight_id").loc[test.fight_id]
    least = raw[["r_prior_ufc_fights", "b_prior_ufc_fights"]].min(axis=1).to_numpy()
    test["experience"] = np.select(
        [least == 0, least < 5],
        ["At least one UFC debutant", "Both returning; minimum below 5"],
        default="Both have at least 5 prior UFC bouts",
    )
    test = test.merge(
        saved[["fight_id", "p_a_wins"]], on="fight_id", validate="one_to_one"
    )
    if len(test) != len(saved) or test.fight_id.nunique() != len(
        frame[frame.split.eq("test")]
    ):
        raise ValueError("Saved predictions do not cover the fixed test cohort")
    test["correct"] = test.p_a_wins.ge(0.5).eq(test.a_won)
    test["confidence"] = np.maximum(test.p_a_wins, 1 - test.p_a_wins)
    prior = development.a_won.mean()
    familiarity = grouped_scores(test, "familiarity", prior)
    experience = grouped_scores(test, "experience", prior)
    failures = []
    for group, part in test.groupby("familiarity"):
        errors = part[~part.correct]
        failures.append(
            dict(
                group=group,
                n=len(part),
                errors=len(errors),
                error_share=len(errors) / (~test.correct).sum(),
                mean_confidence_when_wrong=errors.confidence.mean(),
                errors_with_confidence_at_least_65pct=errors.confidence.ge(0.65).sum(),
            )
        )
    failures = pd.DataFrame(failures)
    unseen = test.familiarity.ne("Both in fitting data")
    exp = experience.set_index("group")
    eligible = familiarity[familiarity.reportable]
    worst = eligible.sort_values("brier_skill").iloc[0]
    largest = failures.sort_values("errors").iloc[-1]
    questions = [
        answer(
            "How often does the held-out model face fighters absent from its fitting sample?",
            f"{int(unseen.sum()):,} of {len(test):,} held-out fights ({unseen.mean():.1%}) include at least one fighter absent from the pre-2023 model-fitting rows.",
            "Familiarity is fixed using IDs in the actual decisive development cohort. An athlete remains absent from the fitting sample even after earlier test appearances update their historical features. This is not the same as a UFC debut or absence from all source history.",
            "deep_generalization_familiarity",
        ),
        answer(
            "How does prediction quality change with the least-experienced opponent?",
            f"Accuracy is {exp.loc['At least one UFC debutant', 'accuracy']:.1%} when at least one fighter is a UFC debutant, versus {exp.loc['Both have at least 5 prior UFC bouts', 'accuracy']:.1%} when both have at least five earlier UFC bouts.",
            "Experience is measured strictly before each fight date. These are diagnostic slices of the already evaluated 2023-2026 model, with 2026 partial. No model, threshold or subgroup is selected for deployment from these results.",
            "deep_generalization_experience",
        ),
        answer(
            "Does the model improve probability quality in every reportable familiarity group?",
            f"The lowest reportable Brier skill is {worst.brier_skill:.1%} for '{worst.group}' (n={int(worst.n)}); {'all reportable groups improve' if eligible.brier_skill.gt(0).all() else 'not all reportable groups improve'} over the fixed development-prevalence forecast.",
            "Brier skill equals one minus model squared probability error divided by reference error on the same fights. A negative value is worse than that constant forecast. The reference prevalence is learned before 2023, not from each subgroup; metrics below 30 fights are withheld.",
            "deep_generalization_familiarity",
        ),
        answer(
            "Which familiarity group contributes the most observed model errors?",
            f"'{largest.group}' contributes {int(largest.errors):,} errors ({largest.error_share:.1%} of all held-out errors); {int(largest.errors_with_confidence_at_least_65pct)} have predicted-winner confidence of at least 65%.",
            "Error volume depends on how often a group occurs, not just its error rate. The 65% threshold is a fixed descriptive cut, not an optimized action rule. Confidence is a model probability and does not establish that a matchup was objectively predictable.",
            "deep_generalization_errors",
        ),
    ]
    audit = test[
        [
            "fight_id",
            "event_id",
            "event_date",
            "a_id",
            "b_id",
            "a_won",
            "p_a_wins",
            "familiarity",
            "experience",
            "correct",
            "confidence",
        ]
    ]
    return {
        "deep_generalization_familiarity": familiarity,
        "deep_generalization_experience": experience,
        "deep_generalization_errors": failures,
        "deep_generalization_predictions": audit,
    }, questions
