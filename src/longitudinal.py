"""Opponent strength and activity research, separate from the frozen test model."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .modeling import NUMERIC, evaluate, model_frame
from .ratings import pre_fight_state
from .research import answer, binomial_summary
from .research_models import expanding_splits
from .statistics import event_bootstrap, wilson_interval


def logistic(columns):
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
    return Pipeline(
        [
            ("transform", transform),
            ("model", LogisticRegression(C=0.3, max_iter=1500, random_state=42)),
        ]
    )


def elo_research(data):
    # Restrict even the state replay to development years, making the boundary explicit.
    fights = data["fights"].loc[data["fights"].fight_year.le(2022)]
    frame = model_frame(fights)
    for k in [16, 32, 64]:
        state = pre_fight_state(fights, k=k).set_index("fight_id").loc[frame.fight_id]
        frame[f"elo_probability_{k}"] = np.where(
            frame.source_red_is_a, state.p_red_elo, 1 - state.p_red_elo
        )
        if k == 32:
            frame["elo_diff"] = (state.r_elo - state.b_elo).to_numpy() * np.where(
                frame.source_red_is_a, 1, -1
            )
    scores, predictions = [], []
    for year, train, validation in expanding_splits(frame):
        probabilities = {
            f"Raw Elo K={k}": validation[f"elo_probability_{k}"].to_numpy()
            for k in [16, 32, 64]
        }
        for name, columns in [
            ("Historical logistic", NUMERIC),
            ("History plus Elo", NUMERIC + ["elo_diff"]),
            ("Elo and division logistic", ["elo_diff"]),
        ]:
            model = logistic(columns).fit(train, train.a_won)
            probabilities[name] = model.predict_proba(validation)[:, 1]
        for name, probability in probabilities.items():
            scores.append(
                dict(
                    year=year,
                    model=name,
                    n=len(validation),
                    train_fights=len(train),
                    **evaluate(validation.a_won, probability),
                )
            )
            rows = validation[
                ["fight_id", "event_id", "event_date", "fight_year", "a_won"]
            ].copy()
            rows["model"], rows["probability"] = name, probability
            predictions.append(rows)
    scores, predictions = (
        pd.DataFrame(scores),
        pd.concat(predictions, ignore_index=True),
    )
    summary = []
    for name, rows in scores.groupby("model"):
        summary.append(
            dict(
                model=name,
                n=rows.n.sum(),
                weighted_log_loss=np.average(rows.log_loss, weights=rows.n),
                weighted_brier=np.average(rows.brier_score, weights=rows.n),
                weighted_accuracy=np.average(rows.accuracy, weights=rows.n),
            )
        )
    summary = pd.DataFrame(summary)
    comparison = predictions.pivot(
        index=["fight_id", "event_id", "a_won"], columns="model", values="probability"
    ).reset_index()

    def loss(probability):
        p = np.clip(probability, 1e-15, 1 - 1e-15)
        return -(comparison.a_won * np.log(p) + (1 - comparison.a_won) * np.log(1 - p))

    improvement = loss(comparison["Historical logistic"]) - loss(
        comparison["History plus Elo"]
    )
    low, high = event_bootstrap(improvement, comparison.event_id)
    gain = pd.DataFrame(
        [
            dict(
                n=len(comparison),
                log_loss_reduction=improvement.mean(),
                low=low,
                high=high,
            )
        ]
    )
    calibration = []
    for name in ["Raw Elo K=32", "History plus Elo"]:
        part = predictions[predictions.model.eq(name)].copy()
        part["band"] = pd.cut(
            part.probability, [0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True
        )
        for band, rows in part.groupby("band", observed=True):
            ci_low, ci_high = wilson_interval(rows.a_won.sum(), len(rows))
            calibration.append(
                dict(
                    model=name,
                    band=str(band),
                    n=len(rows),
                    mean_probability=rows.probability.mean(),
                    observed_win_share=rows.a_won.mean(),
                    low=ci_low,
                    high=ci_high,
                )
            )
    calibration = pd.DataFrame(calibration)
    values = summary.set_index("model")
    losses = scores.pivot(index="year", columns="model", values="log_loss")
    better = (losses["History plus Elo"] < losses["Historical logistic"]).sum()
    raw = values.loc["Raw Elo K=32"]
    sensitivity = values.loc[[f"Raw Elo K={k}" for k in [16, 32, 64]]]
    observed = calibration[
        (calibration.model.eq("Raw Elo K=32")) & calibration.n.ge(50)
    ].copy()
    observed["absolute_gap"] = abs(
        observed.mean_probability - observed.observed_win_share
    )
    widest = observed.sort_values("absolute_gap").iloc[-1]
    questions = [
        answer(
            "Does opponent-adjusted Elo add information beyond the existing histories?",
            f"Adding Elo reduces development log loss by {improvement.mean():+.4f} (paired event-bootstrap 95% interval {low:+.4f} to {high:+.4f}) and improves it in {better} of five annual evaluations.",
            "2018-2022 expanding-year evaluations share identical fights, preprocessing and logistic settings. Positive reduction favors adding Elo. These are already explored development data, not a new independent test; the published holdout model is unchanged.",
            "deep_elo_gain",
        ),
        answer(
            "How informative is an Elo probability on its own?",
            f"The fixed K=32 Elo baseline has {raw.weighted_accuracy:.1%} accuracy, log loss {raw.weighted_log_loss:.4f}, and Brier score {raw.weighted_brier:.4f} across {int(raw.n):,} development evaluation fights.",
            "Every fighter starts at 1500; scale is 400; ratings use all earlier UFC dates including 1993. Draws update with score 0.5 and no contests do not change ratings. There is no division reset, inactivity decay or professional record outside UFC.",
            "deep_elo_summary",
        ),
        answer(
            "How sensitive are raw Elo probabilities to the update speed?",
            f"For K=16, 32 and 64, weighted development log loss ranges from {sensitivity.weighted_log_loss.min():.4f} to {sensitivity.weighted_log_loss.max():.4f}.",
            "All three settings are reported as a sensitivity study. K=32 is the fixed feature specification; the best observed setting is not selected on the 2023-2026 holdout. Faster updates react more strongly to recent results and can change calibration.",
            "deep_elo_folds",
        ),
        answer(
            "Are raw Elo probabilities calibrated across their probability range?",
            f"Among K=32 bins with at least 50 fights, the largest observed calibration gap is {widest.absolute_gap * 100:.1f} points: mean predicted {widest.mean_probability:.1%}, observed {widest.observed_win_share:.1%} (n={int(widest.n)}).",
            "Fixed probability bins describe fighter A's win chance, not confidence in a predicted winner. Wilson intervals ignore recurring fighters. Binning loses detail; this diagnostic does not recalibrate probabilities on the same observations.",
            "deep_elo_calibration",
        ),
    ]
    return {
        "deep_elo_folds": scores,
        "deep_elo_summary": summary,
        "deep_elo_gain": gain,
        "deep_elo_calibration": calibration,
        "deep_elo_predictions": predictions,
    }, questions


def layoff_research(data):
    state = pre_fight_state(data["fights"])
    fights = data["fights"]
    fights = fights[
        fights.in_scope & fights.decisive & fights.fight_year.between(2000, 2025)
    ]
    fights = fights.merge(
        state.drop(columns="event_date"), on="fight_id", validate="one_to_one"
    )
    parts = []
    for corner in ["r", "b"]:
        part = fights[["fight_id", "event_id", "fight_year"]].copy()
        part["fighter_id"] = fights[f"{corner}_id"]
        part["won"] = fights.winner_id.eq(fights[f"{corner}_id"])
        part["days_since"] = fights[f"{corner}_days_since"]
        part["age"] = fights[f"{corner}_age"]
        parts.append(part)
    appearances = pd.concat(parts, ignore_index=True)
    appearances = appearances[appearances.days_since.notna()].copy()
    appearances["rest_band"] = pd.cut(
        appearances.days_since,
        [0, 90, 180, 365, 730, np.inf],
        labels=[
            "1-90 days",
            "91-180 days",
            "181-365 days",
            "366-730 days",
            "Over 730 days",
        ],
    )
    bands = binomial_summary(appearances, "rest_band", "won")
    for band, rows in appearances.groupby("rest_band", observed=True):
        low, high = event_bootstrap(rows.won.astype(float), rows.event_id)
        bands.loc[bands.rest_band.eq(band), ["event_low", "event_high"]] = [low, high]
    pairs = fights[fights.r_days_since.notna() & fights.b_days_since.notna()].copy()
    pairs["rest_gap"] = abs(pairs.r_days_since - pairs.b_days_since)
    pairs = pairs[pairs.rest_gap.ge(90)].copy()
    pairs["longer_rest_won"] = pairs.winner_id.eq(
        pairs.r_id.where(pairs.r_days_since.gt(pairs.b_days_since), pairs.b_id)
    )
    pairs["age_gap"] = abs(pairs.r_age - pairs.b_age)
    pairs["elo_gap"] = abs(pairs.r_elo - pairs.b_elo)
    comparison = []
    for label, rows in [
        ("Rest gap at least 90 days", pairs),
        ("Also age gap at most 2 years", pairs[pairs.age_gap.le(2)]),
        ("Also Elo gap at most 50", pairs[pairs.age_gap.le(2) & pairs.elo_gap.le(50)]),
    ]:
        low, high = event_bootstrap(rows.longer_rest_won.astype(float), rows.event_id)
        comparison.append(
            dict(
                cohort=label,
                n=len(rows),
                longer_rest_wins=rows.longer_rest_won.sum(),
                win_rate=rows.longer_rest_won.mean(),
                event_low=low,
                event_high=high,
            )
        )
    comparison = pd.DataFrame(comparison)
    appearances["age_band"] = pd.cut(
        appearances.age,
        [0, 30, 35, np.inf],
        right=False,
        labels=["Under 30", "30-34", "35 and older"],
    )
    appearances["long_absence"] = appearances.days_since.gt(365)
    ages = binomial_summary(
        appearances[appearances.age.notna()], ["age_band", "long_absence"], "won"
    )
    appearances["period"] = pd.cut(
        appearances.fight_year,
        [1999, 2009, 2019, 2025],
        labels=["2000-2009", "2010-2019", "2020-2025"],
    )
    eras = (
        appearances.groupby("period", observed=True)
        .agg(
            appearances=("fight_id", "size"),
            median_days=("days_since", "median"),
            p90_days=("days_since", lambda x: x.quantile(0.9)),
            over_year_share=("long_absence", "mean"),
        )
        .reset_index()
    )
    short = appearances[~appearances.long_absence]
    long = appearances[appearances.long_absence]
    matched = comparison.iloc[-1]
    older = ages[ages.age_band.eq("35 and older")].set_index("long_absence")
    questions = [
        answer(
            "Do fighters returning after a year win less often?",
            f"Returning fighters with a gap over 365 days win {long.won.mean():.1%} of {len(long):,} appearances, versus {short.won.mean():.1%} of {len(short):,} appearances after shorter gaps.",
            "Decisive UFC bouts in 2000-2025 only; debutants have no UFC rest interval and are excluded. Appearances, not independent fights, are the denominator. A long gap does not identify injury, retirement, suspension or the cause of absence.",
            "deep_layoff_bands",
        ),
        answer(
            "Does the rest-gap association persist among similar-age, similar-rating opponents?",
            f"When rest gaps differ by at least 90 days, age by at most two years and pre-fight Elo by at most 50, the longer-rest fighter wins {matched.win_rate:.1%} of {int(matched.n):,} bouts (event-bootstrap interval {matched.event_low:.1%}-{matched.event_high:.1%}).",
            "These fixed calipers restrict the population and may discard many bouts. Both fighters must have prior UFC appearances. This is a sensitivity check, not random assignment or a fully adjusted causal comparison.",
            "deep_layoff_matched",
        ),
        answer(
            "Is age composition a plausible explanation for the raw inactivity comparison?",
            f"Among returning fighters aged at least 35, win shares are {older.loc[True, 'rate']:.1%} after gaps over one year and {older.loc[False, 'rate']:.1%} after shorter gaps. The age-stratified table shows all three age groups and their sample sizes.",
            "Age strata expose one compositional difference but do not establish mediation or remove strength, era and matchup selection. Wilson intervals are descriptive and ignore recurring fighters.",
            "deep_layoff_age",
        ),
        answer(
            "Has the time between observed UFC appearances changed by era?",
            f"Median return intervals are {eras.iloc[0].median_days:.0f} days in 2000-2009 and {eras.iloc[-1].median_days:.0f} days in 2020-2025; the recent era's 90th percentile is {eras.iloc[-1].p90_days:.0f} days.",
            "These intervals are observed only when a fighter returns for a decisive bout. Athletes who never return are absent, so this is not a time-to-return survival analysis. Earlier non-decisive appearances still reset the previous-fight date.",
            "deep_layoff_eras",
        ),
    ]
    audit = pairs[
        [
            "fight_id",
            "event_id",
            "event_date",
            "r_days_since",
            "b_days_since",
            "rest_gap",
            "age_gap",
            "elo_gap",
            "longer_rest_won",
        ]
    ]
    return {
        "deep_layoff_bands": bands,
        "deep_layoff_matched": comparison,
        "deep_layoff_age": ages,
        "deep_layoff_eras": eras,
        "deep_layoff_pairs": audit,
    }, questions
