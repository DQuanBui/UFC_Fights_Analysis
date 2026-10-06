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


def rate_decomposition(
    frame, group="weight_class", period="period", outcome="is_decision", minimum=50
):
    """Exact within-group and composition decomposition on shared support."""
    cells = frame.groupby([group, period])[outcome].agg(n="size", rate="mean")
    counts = cells.n.unstack(period)
    if set(counts.columns) != {"Before", "After"}:
        raise ValueError("Both Before and After periods are required")
    supported = counts.index[counts.ge(minimum).all(axis=1)]
    counts = counts.loc[supported]
    rates = cells.rate.unstack(period).loc[supported]
    weights = counts / counts.sum()
    result = pd.DataFrame(
        {
            group: supported,
            "n_before": counts.Before,
            "n_after": counts.After,
            "rate_before": rates.Before,
            "rate_after": rates.After,
            "weight_before": weights.Before,
            "weight_after": weights.After,
        }
    ).reset_index(drop=True)
    result["within_contribution"] = (
        (result.rate_after - result.rate_before)
        * (result.weight_before + result.weight_after)
        / 2
    )
    result["mix_contribution"] = (
        (result.weight_after - result.weight_before)
        * (result.rate_before + result.rate_after)
        / 2
    )
    result["total_contribution"] = result.within_contribution + result.mix_contribution
    return result


@chapter("05")
def outcome_questions(data):
    fights, _ = cohort(data)
    period = fights[fights.fight_year.between(2010, 2025)].copy()
    period["period"] = np.where(period.fight_year.lt(2020), "Before", "After")
    decomposition = rate_decomposition(period)
    decisions = fights[fights.is_decision].copy()
    decisions["divided_decision"] = decisions.method.isin(
        ["Decision - Split", "Decision - Majority"]
    )
    divided = binomial_summary(decisions, "weight_class", "divided_decision")
    divided["eligible_comparison"] = divided.n.ge(100)
    scheduled = period[period.standard_format & period.scheduled_rounds.eq(5)]
    title = binomial_summary(scheduled, ["period", "title_fight"], "is_decision")
    prior = (
        data["fights"].query("is_ufc").sort_values(["event_date", "fight_id"]).copy()
    )
    prior["pair"] = prior.apply(lambda r: "|".join(sorted([r.r_id, r.b_id])), axis=1)
    group = prior.groupby("pair")
    prior["previous_winner"] = group.winner_id.shift()
    prior["previous_date"] = group.event_date.shift()
    prior["previous_decisive"] = group.decisive.shift(fill_value=False)
    prior["meeting_number"] = group.cumcount() + 1
    rematch = prior[
        prior.in_scope
        & prior.decisive
        & prior.previous_decisive
        & prior.meeting_number.eq(2)
        & prior.event_date.gt(prior.previous_date)
    ].copy()
    rematch["repeat_winner"] = rematch.winner_id.eq(rematch.previous_winner)
    rematch["gap_years"] = (
        rematch.event_date - rematch.previous_date
    ).dt.days / 365.2425
    rematch["gap_band"] = pd.cut(rematch.gap_years, [0, 1, 3, 100], right=True)
    rematches = binomial_summary(rematch, "gap_band", "repeat_winner")
    biggest = decomposition.sort_values("within_contribution").iloc[0]
    maximum = (
        divided[divided.eligible_comparison]
        .sort_values("rate", ascending=False)
        .iloc[0]
    )
    questions = [
        answer(
            "Which divisions drive the change in decision share?",
            f"On shared divisions, within-division changes contribute {decomposition.within_contribution.sum() * 100:+.2f} percentage points and division-mix changes contribute {decomposition.mix_contribution.sum() * 100:+.2f}. {biggest.weight_class} has the most negative within-division contribution ({biggest.within_contribution * 100:+.2f} points).",
            "The contributions sum exactly to the shared-cohort rate change, which differs from the all-division change. This is a symmetric accounting decomposition, not causal attribution.",
            "deep_decision_decomposition",
        ),
        answer(
            "Where are judges less often unanimous when a fight reaches a decision?",
            f"Among divisions with at least 100 decisions, {maximum.weight_class} has the largest split-or-majority share: {maximum.rate:.1%} of {int(maximum.n)} decisions.",
            "The denominator is decided bouts, not every fight. Split or majority results indicate disagreement in scorecards, not proof of a wrong verdict or bias.",
            "deep_divided_decisions",
        ),
        answer(
            "Do title and non-title fights differ when both have five-round schedules?",
            f"Among standard five-round bouts in 2020-2025, decision shares are {title.loc[title.period.eq('After') & title.title_fight.eq(1), 'rate'].iloc[0]:.1%} for title fights and {title.loc[title.period.eq('After') & title.title_fight.eq(0), 'rate'].iloc[0]:.1%} for non-title fights.",
            "Matching scheduled length and period removes two obvious differences. Division and athlete selection remain uncontrolled, so title status is not a treatment effect.",
            "deep_title_schedule",
        ),
        answer(
            "How often does the first winner win the first UFC rematch?",
            f"The original winner repeats in {rematch.repeat_winner.mean():.1%} of {len(rematch)} eligible first rematches with decisive results in both meetings.",
            "Only the second observed UFC meeting is counted for each unordered pair. Rematches are selectively booked; results do not generalize to all hypothetical rematches.",
            "deep_rematches",
        ),
    ]
    return {
        "deep_decision_decomposition": decomposition,
        "deep_divided_decisions": divided,
        "deep_title_schedule": title,
        "deep_rematches": rematches,
    }, questions


@chapter("06")
def round_questions(data):
    from .statistics import event_bootstrap

    fights, _ = cohort(data)
    rounds = data["rounds"][data["rounds"].fight_id.isin(fights.fight_id)].copy()
    exposure = []
    for schedule, group in fights[fights.standard_format].groupby("scheduled_rounds"):
        for number in range(1, int(schedule) + 1):
            reached = group[group.finish_round.ge(number)]
            seconds = (reached.fight_duration_seconds - 300 * (number - 1)).clip(0, 300)
            endings = (reached.finish_round.eq(number) & reached.is_finish).sum()
            exposure.append(
                dict(
                    scheduled_rounds=int(schedule),
                    round=number,
                    reached=len(reached),
                    finishes=int(endings),
                    observed_minutes=seconds.sum() / 60,
                    finish_risk=endings / len(reached),
                    finishes_per_100_minutes=endings / (seconds.sum() / 60) * 100,
                )
            )
    exposure = pd.DataFrame(exposure)
    complete = fights[
        fights.standard_format
        & fights.scheduled_rounds.eq(3)
        & fights.finish_round.ge(3)
        & fights.round_stats_complete
    ]
    early = rounds[
        rounds.fight_id.isin(complete.fight_id) & rounds.round_no.isin([1, 2])
    ]
    paired = early.pivot(
        index=["fight_id", "fighter_id"], columns="round_no", values="sig_landed"
    ).dropna()
    paired["change_per_minute"] = (paired[2] - paired[1]) / 5
    paired = paired.reset_index().merge(
        fights[["fight_id", "event_id"]], on="fight_id", validate="many_to_one"
    )
    low, high = event_bootstrap(
        paired.change_per_minute.to_numpy(), paired.event_id.to_numpy()
    )
    activity = pd.DataFrame(
        [
            dict(
                fighter_pairs=len(paired),
                fights=paired.fight_id.nunique(),
                round1_rate=paired[1].mean() / 5,
                round2_rate=paired[2].mean() / 5,
                paired_change=paired.change_per_minute.mean(),
                event_bootstrap_low=low,
                event_bootstrap_high=high,
            )
        ]
    )
    continuing = fights[
        fights.standard_format
        & fights.decisive
        & fights.finish_round.ge(2)
        & fights.round_stats_complete
    ]
    opening = rounds[rounds.round_no.eq(1)].merge(
        continuing[["fight_id", "r_id", "winner_id", "event_id"]],
        on="fight_id",
        validate="many_to_one",
    )
    red = opening[opening.fighter_id.eq(opening.r_id)].set_index("fight_id")
    blue = (
        opening[opening.fighter_id.ne(opening.r_id)]
        .set_index("fight_id")
        .reindex(red.index)
    )
    leads = []
    for metric in ["sig_landed", "td_success", "ctrl_seconds"]:
        delta = red[metric] - blue[metric]
        usable = delta.notna() & delta.ne(0)
        leaders = red.fighter_id.where(delta.gt(0), blue.fighter_id)
        won = leaders.loc[usable].eq(red.loc[usable, "winner_id"])
        low, high = wilson_interval(won.sum(), len(won))
        leads.append(
            dict(
                round1_metric=metric,
                eligible_fights=len(won),
                leader_wins=won.sum(),
                leader_win_rate=won.mean(),
                low=low,
                high=high,
                excluded_ties_or_missing=len(red) - len(won),
            )
        )
    leads = pd.DataFrame(leads)
    control = (
        rounds.assign(known_control=rounds.ctrl_seconds.notna())
        .groupby("fight_year")
        .agg(
            fighter_rounds=("fight_id", "size"),
            known_control_share=("known_control", "mean"),
        )
        .reset_index()
    )
    strike = leads[leads.round1_metric.eq("sig_landed")].iloc[0]
    recent = control[control.fight_year.ge(2010)]
    questions = [
        answer(
            "Is later-round finishing still lower after accounting for time exposed?",
            "For standard three-round bouts, finish incidence per 100 observed fight-minutes is "
            + ", ".join(
                f"round {int(r['round'])}: {r['finishes_per_100_minutes']:.2f}"
                for _, r in exposure[exposure.scheduled_rounds.eq(3)].iterrows()
            )
            + ".",
            "Incidence uses actual elapsed exposure, while conditional risk uses fights reaching the round. Neither removes survivor selection or gives an individual instantaneous hazard.",
            "deep_round_exposure",
        ),
        answer(
            "Do the same fighters change striking pace between full rounds?",
            f"Among {paired.fight_id.nunique():,} standard three-round bouts reaching round 3, round-2 striking changes by {activity.paired_change.iloc[0]:+.3f} landed strikes per minute versus round 1 (event-bootstrap interval {activity.event_bootstrap_low.iloc[0]:+.3f} to {activity.event_bootstrap_high.iloc[0]:+.3f}).",
            "Both rounds are full five-minute exposures for the same fighter-bouts. This removes between-round composition differences but restricts the conclusion to bouts surviving two rounds.",
            "deep_paired_round_pace",
        ),
        answer(
            "How informative is leading the first round when the fight continues?",
            f"The first-round significant-strike leader wins {strike.leader_win_rate:.1%} of {int(strike.eligible_fights):,} eligible continuing bouts.",
            "This is a within-fight descriptive association, not a pre-fight model feature or a claim about judges awarding round 1. Ties and missing measures are excluded separately for each metric.",
            "deep_opening_round_leads",
        ),
        answer(
            "Can control-time trends be compared throughout UFC history?",
            f"Annual known-control coverage spans {control.known_control_share.min():.1%}-{control.known_control_share.max():.1%}; from 2010 onward the minimum is {recent.known_control_share.min():.1%}.",
            "Missing historical control is not evidence that fighters did not control opponents. Restrict comparisons to years with adequate coverage and show observation counts.",
            "deep_control_coverage",
        ),
    ]
    return {
        "deep_round_exposure": exposure,
        "deep_paired_round_pace": activity,
        "deep_opening_round_leads": leads,
        "deep_control_coverage": control,
    }, questions


@chapter("07")
def adjusted_associations(data):
    from scipy.stats import pearsonr
    from sklearn.linear_model import LogisticRegression

    from .modeling import model_frame

    fights, appearances = cohort(data)
    frame = model_frame(data["fights"])
    frame = frame[frame.fight_year.between(2000, 2025)].copy()
    extras = fights.set_index("fight_id")
    sign = np.where(frame.source_red_is_a, 1, -1)
    frame["reach_2in"] = (
        extras.loc[frame.fight_id, "reach_inches_difference"].to_numpy() * sign / 2
    )
    frame["height_2in"] = (
        extras.loc[frame.fight_id, "height_inches_difference"].to_numpy() * sign / 2
    )
    frame["age_5yr"] = frame.age_diff / 5
    frame["experience_5fights"] = frame.prior_ufc_fights_diff / 5
    variables = ["age_5yr", "height_2in", "reach_2in", "experience_5fights"]
    before = len(frame)
    frame = frame.dropna(subset=variables)
    adequate = frame.weight_class.value_counts().loc[lambda x: x.ge(100)].index
    frame = frame[frame.weight_class.isin(adequate)].copy()
    frame["period"] = (frame.fight_year // 10).astype(str)
    design = pd.concat(
        [
            frame[variables],
            pd.get_dummies(
                frame[["weight_class", "period"]], drop_first=True, dtype=float
            ),
        ],
        axis=1,
    )
    model = LogisticRegression(C=1e6, max_iter=1000, tol=1e-7)
    model.fit(design, frame.a_won)
    estimates = model.coef_[0][: len(variables)]
    codes, events = pd.factorize(frame.event_id)
    rng = np.random.default_rng(42)
    sampled = []
    for _ in range(250):
        event_counts = rng.multinomial(
            len(events), np.full(len(events), 1 / len(events))
        )
        model.fit(design, frame.a_won, sample_weight=event_counts[codes])
        sampled.append(model.coef_[0][: len(variables)])
    low, high = np.quantile(np.asarray(sampled), [0.025, 0.975], axis=0)
    adjusted = pd.DataFrame(
        {
            "feature": variables,
            "log_odds_coefficient": estimates,
            "odds_ratio": np.exp(estimates),
            "event_bootstrap_low": np.exp(low),
            "event_bootstrap_high": np.exp(high),
            "fights": len(frame),
            "excluded_from_initial_window": before - len(frame),
        }
    )
    crude = LogisticRegression(C=1e6, max_iter=1000, tol=1e-7).fit(
        frame[["reach_2in"]], frame.a_won
    )
    adjusted["bootstrap_repetitions"] = 250
    reach = adjusted[adjusted.feature.eq("reach_2in")].iloc[0]
    mode = (
        appearances.groupby(["fighter_id", "weight_class"])
        .size()
        .rename("bouts")
        .reset_index()
        .sort_values(
            ["fighter_id", "bouts", "weight_class"], ascending=[True, False, True]
        )
        .drop_duplicates("fighter_id")
    )
    physical = (
        appearances.drop_duplicates("fighter_id")[
            ["fighter_id", "height_inches", "reach_inches"]
        ]
        .merge(
            mode[["fighter_id", "weight_class"]], on="fighter_id", validate="one_to_one"
        )
        .dropna()
    )
    demeaned = physical[["height_inches", "reach_inches"]] - physical.groupby(
        "weight_class"
    )[["height_inches", "reach_inches"]].transform("mean")
    correlation = pd.DataFrame(
        [
            dict(
                fighters=len(physical),
                pooled_correlation=pearsonr(
                    physical.height_inches, physical.reach_inches
                ).statistic,
                within_modal_division_correlation=pearsonr(
                    demeaned.height_inches, demeaned.reach_inches
                ).statistic,
            )
        ]
    )
    sensitivity = []
    for max_gap in [2, 5, 65]:
        subset = fights[
            fights.decisive
            & fights.reach_inches_difference.notna()
            & fights.reach_inches_difference.ne(0)
            & fights.age_difference.notna()
            & fights.age_difference.abs().le(max_gap)
        ]
        won = subset.winner_id.eq(
            subset.r_id.where(subset.reach_inches_difference.gt(0), subset.b_id)
        )
        interval_low, interval_high = wilson_interval(won.sum(), len(won))
        sensitivity.append(
            dict(
                maximum_age_gap=max_gap,
                fights=len(won),
                longer_reach_win_rate=won.mean(),
                low=interval_low,
                high=interval_high,
            )
        )
    sensitivity = pd.DataFrame(sensitivity)
    age = adjusted[adjusted.feature.eq("age_5yr")].iloc[0]
    questions = [
        answer(
            "Does the reach association survive adjustment for other fighter characteristics?",
            f"On the same {len(frame):,} complete-case bouts, the crude odds ratio per additional 2 inches of reach is {np.exp(crude.coef_[0, 0]):.3f}; the adjusted ratio is {reach.odds_ratio:.3f} (250-event-bootstrap 95% interval {reach.event_bootstrap_low:.3f}-{reach.event_bootstrap_high:.3f}).",
            "The exploratory logistic model includes age, height and prior UFC experience differences plus division and decade indicators. Measurements are snapshot traits; missingness, matchup quality and recurring fighters prevent a causal interpretation. Odds ratios are not percentage-point changes.",
            "deep_adjusted_attributes",
        ),
        answer(
            "Does age still have an association after allowing for UFC experience?",
            f"A five-year age difference has adjusted odds ratio {age.odds_ratio:.3f} (interval {age.event_bootstrap_low:.3f}-{age.event_bootstrap_high:.3f}) for the older fighter, holding the included covariates fixed.",
            "Age and experience are not interchangeable. Linear log-odds, selected complete cases, mild numerical regularization (C=1,000,000) and residual confounding limit the result.",
            "deep_adjusted_attributes",
        ),
        answer(
            "How much of the height-reach correlation reflects different divisions?",
            f"Among {len(physical):,} unique fighters, pooled Pearson correlation is {correlation.pooled_correlation.iloc[0]:.3f}; after removing modal-division means it is {correlation.within_modal_division_correlation.iloc[0]:.3f}.",
            "Each fighter appears once and is assigned their most-observed division. The residual correlation measures within-group linear association; athletes competing in multiple divisions make this grouping approximate.",
            "deep_within_division_correlation",
        ),
        answer(
            "What happens when comparing reach among similarly aged opponents?",
            f"For opponents within two years of age, the longer-reach fighter wins {sensitivity.iloc[0].longer_reach_win_rate:.1%} across {int(sensitivity.iloc[0].fights):,} bouts.",
            "An age caliper is a transparent sensitivity check, not full matching. It changes the eligible population and still leaves division, skill and height differences.",
            "deep_reach_age_caliper",
        ),
    ]
    return {
        "deep_adjusted_attributes": adjusted,
        "deep_within_division_correlation": correlation,
        "deep_reach_age_caliper": sensitivity,
    }, questions


@chapter("08")
def model_robustness(data):
    from .research_models import feature_ablation, holdout_diagnostics

    folds = feature_ablation(data["fights"])
    summaries = []
    for name, rows in folds.groupby("features"):
        summaries.append(
            dict(
                features=name,
                validation_fights=rows.n.sum(),
                weighted_accuracy=np.average(rows.accuracy, weights=rows.n),
                weighted_log_loss=np.average(rows.log_loss, weights=rows.n),
                weighted_mean_fold_auc=np.average(rows.roc_auc, weights=rows.n),
            )
        )
    summary = pd.DataFrame(summaries)
    metadata = json.loads((TABLES / "model_metadata.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(TABLES / "test_predictions.csv")
    predictions = predictions[predictions.model.eq(metadata["selected_model"])]
    confidence, years, gains = holdout_diagnostics(data["fights"], predictions)
    by_name = summary.set_index("features")
    full = by_name.loc["Full historical features"]
    basic = by_name.loc["Age and division"]
    gain = gains[gains.comparator.eq("Source red-corner heuristic")].iloc[0]
    questions = [
        answer(
            "Do richer histories help consistently before the held-out test period?",
            f"Across five expanding-year evaluations in 2018-2022, full historical features have weighted log loss {full.weighted_log_loss:.4f}, compared with {basic.weighted_log_loss:.4f} for age and division alone.",
            "The same logistic specification is used for each feature family and preprocessing is refitted inside each chronological fold. Inspect year-specific results for consistency. These are exploratory development checks, not a newly untouched benchmark.",
            "deep_feature_ablation_summary",
        ),
        answer(
            "How stable is the selected model across held-out years?",
            f"Annual accuracy ranges from {years.accuracy.min():.1%} to {years.accuracy.max():.1%} in the existing 2023-2026 holdout.",
            "These are diagnostics of the already evaluated model, not a reason to select a different model on the same holdout. The final year has fewer months and fights.",
            "deep_model_years",
        ),
        answer(
            "Are the model’s most confident predictions more reliable?",
            f"The populated confidence bands have observed accuracy from {confidence.accuracy.min():.1%} to {confidence.accuracy.max():.1%}; the highest populated band contains {int(confidence.iloc[-1].n)} fights.",
            "Compare mean confidence with observed accuracy and interval width. No confidence cutoff is optimized on these test data; small high-confidence groups can be noisy.",
            "deep_model_confidence",
        ),
        answer(
            "Does the model improve on always choosing the source red corner?",
            f"The selected model improves accuracy by {gain.accuracy_gain * 100:.2f} percentage points on the same fights; the paired event-bootstrap interval is {gain.event_bootstrap_low * 100:.2f} to {gain.event_bootstrap_high * 100:.2f} points.",
            "Pairing comparisons within the same fight is more informative than comparing two separate accuracy intervals. Recurring fighters across events still create unmodeled dependence.",
            "deep_paired_model_gain",
        ),
    ]
    return {
        "deep_feature_ablation_folds": folds,
        "deep_feature_ablation_summary": summary,
        "deep_model_confidence": confidence,
        "deep_model_years": years,
        "deep_paired_model_gain": gains,
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
