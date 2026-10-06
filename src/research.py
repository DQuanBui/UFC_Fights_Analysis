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
