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


def product_decomposition(count_before, count_after, rate_before, rate_after):
    """Exact symmetric decomposition of a change in count times rate."""
    return (
        (count_after - count_before) * (rate_before + rate_after) / 2,
        (rate_after - rate_before) * (count_before + count_after) / 2,
    )


@chapter("03")
def growth_drivers(data):
    from .analysis import annual_summary

    fights, appearances = cohort(data)
    annual = annual_summary(fights, appearances).set_index("fight_year")
    changes = []
    for start, end in [(2005, 2010), (2010, 2015), (2015, 2025)]:
        a, b = annual.loc[start], annual.loc[end]
        event_effect, card_effect = product_decomposition(
            a.events, b.events, a.fights_per_event, b.fights_per_event
        )
        changes.append(
            dict(
                start=start,
                end=end,
                fight_change=b.fights - a.fights,
                event_count_contribution=event_effect,
                card_size_contribution=card_effect,
                start_events=a.events,
                end_events=b.events,
                start_card_size=a.fights_per_event,
                end_card_size=b.fights_per_event,
            )
        )
    changes = pd.DataFrame(changes)
    events = fights[fights.fight_year.le(2025)].drop_duplicates("event_id")
    concentration = []
    for year, rows in events.groupby("fight_year"):
        shares = rows.country.value_counts(normalize=True)
        concentration.append(
            dict(
                year=year,
                events=len(rows),
                countries=len(shares),
                country_hhi=(shares**2).sum(),
                effective_country_count=1 / (shares**2).sum(),
                usa_share=rows.country.eq("USA").mean(),
                largest_city_share=rows.city.value_counts(normalize=True).iloc[0],
            )
        )
    concentration = pd.DataFrame(concentration)
    first_year = (
        data["appearances"].query("is_ufc").groupby("fighter_id").fight_year.min()
    )
    participation = appearances[["fight_year", "fighter_id"]].drop_duplicates()
    participation["first_observed_year"] = participation.fighter_id.map(first_year)
    participation["new_entrant"] = participation.fight_year.eq(
        participation.first_observed_year
    )
    entrants = (
        participation.groupby("fight_year")
        .agg(
            active_fighters=("fighter_id", "size"), new_fighters=("new_entrant", "sum")
        )
        .reset_index()
    )
    entrants["returning_fighters"] = entrants.active_fighters - entrants.new_fighters
    entrants["entrant_share"] = entrants.new_fighters / entrants.active_fighters
    retention = []
    for year in range(2000, 2025):
        ids = set(participation.loc[participation.fight_year.eq(year), "fighter_id"])
        following = set(
            participation.loc[participation.fight_year.eq(year + 1), "fighter_id"]
        )
        retention.append(
            dict(
                year=year,
                active=len(ids),
                seen_next_year=len(ids & following),
                next_year_return_share=len(ids & following) / len(ids),
            )
        )
    retention = pd.DataFrame(retention)
    row = changes.iloc[1]
    geo = concentration.set_index("year")
    last = entrants.query("fight_year == 2025").iloc[0]
    questions = [
        answer(
            "Did fight volume grow through more events or larger cards?",
            f"From 2010 to 2015, fight volume changed by {row.fight_change:.0f}; the exact decomposition attributes {row.event_count_contribution:.1f} fights to event count and {row.card_size_contribution:.1f} to card size.",
            "This symmetric accounting identity divides the interaction equally. It explains arithmetic contributions, not causal drivers of the schedule.",
            "deep_growth_decomposition",
        ),
        answer(
            "Does visiting more countries mean events are evenly distributed?",
            f"In 2025 the snapshot includes {int(geo.loc[2025, 'countries'])} country/territory labels, but concentration is equivalent to only {geo.loc[2025, 'effective_country_count']:.2f} equally represented hosts.",
            "The reciprocal Herfindahl index distinguishes geographic breadth from geographic balance. It weights events, not fighters or audience reach.",
            "deep_geographic_concentration",
        ),
        answer(
            "How much of the active roster consists of new UFC entrants?",
            f"In 2025, {int(last.new_fighters)} of {int(last.active_fighters)} active fighters ({last.entrant_share:.1%}) had their first observed UFC bout that year.",
            "A first observed appearance is not necessarily a professional debut. Annual counts omit inactive athletes, so this is participation rather than roster size.",
            "deep_entrant_flow",
        ),
        answer(
            "How persistent is yearly fighter participation?",
            f"Among {int(retention.iloc[-1].active)} fighters active in 2024, {int(retention.iloc[-1].seen_next_year)} ({retention.iloc[-1].next_year_return_share:.1%}) also appear in 2025.",
            "Not returning the next year does not prove release or retirement. Injury, inactivity and incomplete source coverage can produce the same pattern; partial 2026 is excluded.",
            "deep_annual_return",
        ),
    ]
    return {
        "deep_growth_decomposition": changes,
        "deep_geographic_concentration": concentration,
        "deep_entrant_flow": entrants,
        "deep_annual_return": retention,
    }, questions


@chapter("04")
def career_questions(data):
    fights, appearances = cohort(data)
    decisive = appearances[appearances.decisive].copy()
    record = binomial_summary(decisive, ["fighter_id", "fighter_name"], "won")
    leaders = []
    for minimum in [5, 10, 20]:
        group = (
            record[record.n.ge(minimum)]
            .sort_values(["rate", "n"], ascending=False)
            .head(10)
            .copy()
        )
        group["minimum_decisive_fights"] = minimum
        leaders.append(group)
    leaders = pd.concat(leaders, ignore_index=True)
    rows = fights[fights.decisive & fights.fight_year.ge(2000)].copy()
    rookie_r = rows.r_prior_ufc_fights.eq(0) & rows.b_prior_ufc_fights.ge(5)
    rookie_b = rows.b_prior_ufc_fights.eq(0) & rows.r_prior_ufc_fights.ge(5)
    matchups = rows[rookie_r | rookie_b].copy()
    matchups["debutant_won"] = matchups.winner_id.eq(
        matchups.r_id.where(rookie_r, matchups.b_id)
    )
    matchups["period"] = np.where(
        matchups.fight_year.lt(2010),
        "2000-2009",
        np.where(matchups.fight_year.lt(2020), "2010-2019", "2020-2026"),
    )
    debut = binomial_summary(matchups, "period", "debutant_won")
    age = fights[
        fights.decisive & fights.age_difference.notna() & fights.age_difference.ne(0)
    ].copy()
    age["age_gap"] = pd.cut(
        age.age_difference.abs(), [0, 2, 5, 10, 65], include_lowest=True
    )
    age["younger_won"] = age.winner_id.eq(
        age.r_id.where(age.age_difference.lt(0), age.b_id)
    )
    age = binomial_summary(age, "age_gap", "younger_won")
    all_ufc = (
        data["appearances"].query("is_ufc").sort_values(["event_date", "fight_id"])
    )
    end = fights.event_date.max()
    careers = []
    for fid, group in all_ufc.groupby("fighter_id"):
        date = group.event_date.min()
        opening = group[group.event_date.eq(date)]
        if (
            date.year < 2000
            or date > end - pd.Timedelta(days=730)
            or len(opening) != 1
            or not opening.decisive.iloc[0]
        ):
            continue
        window = group[group.event_date.le(date + pd.Timedelta(days=730))]
        careers.append(
            dict(
                fighter_id=fid,
                debut_date=date,
                debut_win=bool(opening.won.iloc[0]),
                fights_within_730_days=len(window),
                reached_three_fights=len(window) >= 3,
            )
        )
    careers = pd.DataFrame(careers)
    continuation = binomial_summary(careers, "debut_win", "reached_three_fights")
    continuation["mean_fights"] = (
        careers.groupby("debut_win")
        .fights_within_730_days.mean()
        .reindex(continuation.debut_win)
        .to_numpy()
    )
    top = record[record.n.ge(10)].sort_values(["rate", "n"], ascending=False).iloc[0]
    rate = matchups.debutant_won.mean()
    questions = [
        answer(
            "How certain are the highest win-rate rankings?",
            f"With a 10-decisive-fight minimum, {top.fighter_name} leads at {int(top.successes)}/{int(top.n)} ({top.rate:.1%}); the descriptive Wilson interval is {top.low:.1%}-{top.high:.1%}.",
            "The table shows how leaders change at 5, 10 and 20 fights. Intervals overlap and ignore opponent quality and athlete dependence; the ranking is not a definitive skill ordering.",
            "deep_ranking_stability",
        ),
        answer(
            "How do UFC newcomers fare against established UFC opponents?",
            f"In {len(matchups):,} decisive matchups since 2000 with exactly one newcomer and an opponent with at least five prior UFC bouts, the newcomer wins {rate:.1%}.",
            "Prior UFC experience is measured before the date. Newcomers may have extensive experience elsewhere, and matchmaking is selective.",
            "deep_debut_matchups",
        ),
        answer(
            "Does the age association strengthen as the age gap widens?",
            f"Younger-fighter win shares range from {age.rate.min():.1%} to {age.rate.max():.1%} across the four unequal-age bands shown below.",
            "Compare the bands and their sample sizes rather than assuming a linear age effect. Equal-age and missing-DOB bouts are excluded.",
            "deep_age_gap",
        ),
        answer(
            "Is a successful UFC debut associated with more early opportunities?",
            f"Among debutants with 730 observable follow-up days, {continuation.loc[continuation.debut_win, 'rate'].iloc[0]:.1%} of debut winners and {continuation.loc[~continuation.debut_win, 'rate'].iloc[0]:.1%} of debut losers reach three observed UFC bouts within that window.",
            "Every entrant has the same follow-up horizon. This is observed participation, not a retention contract or causal effect; recent censored entrants and ambiguous same-day debuts are excluded.",
            "deep_career_continuation",
        ),
    ]
    return {
        "deep_ranking_stability": leaders,
        "deep_debut_matchups": debut,
        "deep_age_gap": age,
        "deep_career_continuation": continuation,
        "deep_career_followup": careers,
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
