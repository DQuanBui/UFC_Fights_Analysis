"""Question-led research extensions with computed answers and explicit caveats."""

import argparse
import json

import numpy as np
import pandas as pd

from .analysis import cohort, save_tables
from .data_loader import TABLES, load_raw
from .statistics import wilson_interval

CHAPTERS = {}


def chapter(number):
    def register(function):
        CHAPTERS[number] = function
        return function

    return register


def answer(question, evidence, interpretation, table):
    return dict(
        question=question, answer=evidence, interpretation=interpretation, table=table
    )


def binomial_summary(frame, groups, outcome):
    result = (
        frame.groupby(groups, observed=True, dropna=False)[outcome]
        .agg(n="size", successes="sum")
        .reset_index()
    )
    result["rate"] = result.successes / result.n
    intervals = [wilson_interval(k, n) for k, n in zip(result.successes, result.n)]
    result[["low", "high"]] = pd.DataFrame(intervals, index=result.index)
    return result


@chapter("01")
def source_bias(data):
    """Quantify promotion contamination, selection, and unobserved statistics."""
    all_fights = data["fights"]
    fights, appearances = cohort(data)
    scope = all_fights.assign(
        scope=np.where(
            all_fights.in_scope,
            "UFC 1994-2026",
            np.where(
                all_fights.is_ufc,
                "UFC outside reporting window",
                "Other promotion / feeder",
            ),
        )
    )
    scope = (
        scope.groupby("scope")
        .agg(
            fights=("fight_id", "size"),
            events=("event_id", "nunique"),
            finish_rate=("is_finish", "mean"),
            complete_stats=("round_stats_complete", "mean"),
        )
        .reset_index()
    )
    missing = appearances.assign(
        period=np.where(
            appearances.fight_year.le(1999),
            "1994-1999",
            np.where(
                appearances.fight_year.le(2009),
                "2000-2009",
                np.where(appearances.fight_year.le(2019), "2010-2019", "2020-2026"),
            ),
        )
    )
    missing = (
        missing.groupby("period")
        .agg(
            appearances=("fight_id", "size"),
            age_missing=("age", lambda s: s.isna().mean()),
            reach_missing=("reach_inches", lambda s: s.isna().mean()),
            control_missing=("ctrl_seconds", lambda s: s.isna().mean()),
            statistics_missing=("sig_landed", lambda s: s.isna().mean()),
        )
        .reset_index()
    )
    selection = fights.assign(stats_observed=fights.round_stats_complete)
    selection = (
        selection.groupby("stats_observed")
        .agg(
            fights=("fight_id", "size"),
            finish_rate=("is_finish", "mean"),
            mean_year=("fight_year", "mean"),
            duration_minutes=("fight_duration_seconds", lambda s: s.mean() / 60),
        )
        .reset_index()
    )
    raw = load_raw()
    master = raw["master"].set_index("fight_id")
    scoped_master = master.loc[fights.fight_id]
    totals = appearances.groupby("fight_id").sig_landed.sum(min_count=2)
    observed = totals.dropna()
    zero_bias = pd.DataFrame(
        [
            dict(
                scope="Selected UFC fights",
                all_fights=len(fights),
                observed_fights=len(observed),
                naive_master_mean=(
                    scoped_master.r_total_sig_landed + scoped_master.b_total_sig_landed
                ).mean(),
                observed_mean=observed.mean(),
                missing_fights=totals.isna().sum(),
            )
        ]
    )
    outside = int((~all_fights.in_scope).sum())
    questions = [
        answer(
            "How much would using the complete master change the UFC population?",
            f"It would add {outside:,} fights ({outside / len(all_fights):.1%} of all source fights) beyond the {len(fights):,}-fight reporting cohort.",
            "This is a scope mismatch, not additional UFC coverage. Non-UFC results must not enter UFC counts or prior-UFC records.",
            "deep_scope_bias",
        ),
        answer(
            "Are missing round statistics a random subset of UFC fights?",
            f"{int((~fights.round_stats_complete).sum())} scoped fights lack statistics; their mean event year is {selection.loc[~selection.stats_observed, 'mean_year'].iloc[0]:.1f}.",
            "Coverage differs historically. Modern activity cannot be fairly compared with early raw totals without displaying observation rates.",
            "deep_missing_selection",
        ),
        answer(
            "How much does zero-filling missing fights change average striking totals?",
            f"The source master gives {zero_bias.naive_master_mean.iloc[0]:.2f} combined significant strikes per scoped fight; the complete observed subset gives {zero_bias.observed_mean.iloc[0]:.2f}.",
            "The aggregate difference is small because scoped missingness is rare, but early-period comparisons can be affected much more. Missing control is a separate coverage problem.",
            "deep_zero_bias",
        ),
    ]
    return {
        "deep_scope_bias": scope,
        "deep_missingness": missing,
        "deep_missing_selection": selection,
        "deep_zero_bias": zero_bias,
    }, questions


@chapter("02")
def cleaning_sensitivity(data):
    """How do timing conventions, scope and denominator rules change answers?"""
    from .cleaning import parse_clock

    fights, appearances = cohort(data)
    naive = (fights.finish_round - 1) * 300 + fights.finish_time.map(parse_clock)
    timing = fights[["time_format", "fight_duration_seconds"]].copy()
    timing["naive_seconds"] = naive
    timing["absolute_error"] = abs(naive - timing.fight_duration_seconds)
    timing = (
        timing.groupby("time_format")
        .agg(
            fights=("absolute_error", "size"),
            affected=("absolute_error", lambda s: s.gt(0).sum()),
            mean_error_seconds=("absolute_error", "mean"),
            maximum_error_seconds=("absolute_error", "max"),
        )
        .reset_index()
    )
    variants = []
    for name, rows in [
        ("All scoped fights", fights),
        ("Decisive fights only", fights[fights.decisive]),
        ("Complete statistics only", fights[fights.round_stats_complete]),
        ("Standard schedules only", fights[fights.standard_format]),
        (
            "2000-2025 complete calendar years",
            fights[fights.fight_year.between(2000, 2025)],
        ),
    ]:
        variants.append(
            dict(
                definition=name,
                fights=len(rows),
                finish_rate=rows.is_finish.mean(),
                decision_rate=rows.is_decision.mean(),
                duration_minutes=rows.fight_duration_seconds.mean() / 60,
            )
        )
    variants = pd.DataFrame(variants)
    observed = appearances[
        appearances.sig_landed.notna() & appearances.fight_duration_seconds.gt(0)
    ]
    rates = (
        observed.groupby("weight_class")
        .apply(
            lambda rows: pd.Series(
                dict(
                    fighter_fights=len(rows),
                    equal_fight_weighted_rate=rows.sig_per_minute.mean(),
                    exposure_weighted_rate=rows.sig_landed.sum()
                    / (rows.fight_duration_seconds.sum() / 60),
                )
            ),
            include_groups=False,
        )
        .reset_index()
    )
    rates["difference"] = rates.equal_fight_weighted_rate - rates.exposure_weighted_rate
    zero_attempts = (
        appearances.groupby("weight_class")
        .agg(
            appearances=("fight_id", "size"),
            no_strike_attempts=("sig_atmp", lambda s: s.eq(0).sum()),
            no_takedown_attempts=("td_atmp", lambda s: s.eq(0).sum()),
            mean_observed_td_accuracy=("td_accuracy", "mean"),
        )
        .reset_index()
    )
    corrected = int(timing.affected.sum())
    gap = (variants.iloc[1].finish_rate - variants.iloc[0].finish_rate) * 100
    eligible = (
        rates[rates.fighter_fights.ge(100)]
        .sort_values("difference", ascending=False)
        .iloc[0]
    )
    questions = [
        answer(
            "How many UFC durations would a universal five-minute formula misstate?",
            f"{corrected} fights have different durations under the naive formula; the largest error is {timing.maximum_error_seconds.max():.0f} seconds.",
            "Historical schedules require their documented round lengths. The schedule-specific audit shows which eras and formats are affected.",
            "deep_timing_sensitivity",
        ),
        answer(
            "Does excluding unresolved outcomes change the headline finish rate?",
            f"Restricting to decisive fights changes the finish share by {gap:+.2f} percentage points, from {variants.iloc[0].finish_rate:.2%} to {variants.iloc[1].finish_rate:.2%}.",
            "Both denominators can answer a question, but they cannot share the same label. Reporting only complete statistics also changes the cohort.",
            "deep_cohort_sensitivity",
        ),
        answer(
            "Are average fight rates interchangeable with exposure-weighted rates?",
            f"Among divisions with at least 100 fighter-fights, {eligible.weight_class} has the largest positive gap: {eligible.equal_fight_weighted_rate:.2f} versus {eligible.exposure_weighted_rate:.2f} significant strikes per minute.",
            "Equal-fight averages give a very short bout the same weight as a full decision. Exposure-weighted rates estimate production per observed minute. Neither should be presented as the other.",
            "deep_rate_weighting",
        ),
        answer(
            "Should a fight with no takedown attempts have zero takedown accuracy?",
            f"{int(zero_attempts.no_takedown_attempts.sum()):,} fighter-fight records have zero takedown attempts.",
            "Accuracy is undefined with zero attempts. Filling it with zero would mix tactical non-participation with failed attempts.",
            "deep_zero_attempts",
        ),
    ]
    return {
        "deep_timing_sensitivity": timing,
        "deep_cohort_sensitivity": variants,
        "deep_rate_weighting": rates,
        "deep_zero_attempts": zero_attempts,
    }, questions


def run_chapter(number, data=None):
    if data is None:
        from .pipeline import prepare

        data = prepare(False)
    tables, answers = CHAPTERS[number](data)
    save_tables(tables)
    (TABLES / f"research_{number}.json").write_text(
        json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return tables, answers


def display_answers(answers):
    from IPython.display import Markdown, display

    for item in answers:
        display(
            Markdown(
                f"### {item['question']}\n\n**Answer.** {item['answer']}\n\n**Interpretation.** {item['interpretation']}"
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", choices=sorted(CHAPTERS))
    selected = parser.parse_args().chapter
    tables, answers = run_chapter(selected)
    for item in answers:
        print(item["question"], item["answer"])
