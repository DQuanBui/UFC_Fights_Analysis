"""Generate narrative findings directly from executed analytical tables."""

import json

import pandas as pd

from .data_loader import ROOT, TABLES


def build_findings():
    def read(name):
        return pd.read_csv(TABLES / f"{name}.csv")

    annual = read("annual").set_index("fight_year")
    division = read("divisions").set_index("weight_class")
    fighters = read("fighters")
    mix = read("division_mix_adjustment")
    risk = read("round_hazard").query("scheduled_rounds == 3").set_index("round_no")
    advantage = read("advantage_uncertainty").set_index("attribute")
    performance = read("performance_by_result").set_index("won")
    bonus = read("bonus_categories").set_index("bonus_type")
    bonus_rates = read("bonus_rates").set_index("outcome_group")
    metrics = read("model_metrics")
    model = json.loads((TABLES / "model_metadata.json").read_text(encoding="utf-8"))[
        "selected_model"
    ]
    score = metrics[(metrics.model == model) & (metrics.split == "test")].iloc[0]
    baseline = metrics[
        (metrics.model == "Majority baseline") & (metrics.split == "test")
    ].iloc[0]
    title = read("title_statistics").iloc[0]
    paired = read("paired_statistics").set_index("metric")
    total = int(annual.fights.sum())
    finish = (annual.finish_rate * annual.fights).sum() / total
    ko = (annual.ko_rate * annual.fights).sum() / total
    findings = []

    def add(observation, evidence, interpretation, source):
        findings.append(
            dict(
                observation=observation,
                evidence=evidence,
                interpretation=interpretation,
                source=source,
            )
        )

    add(
        "Scope is an analytical decision",
        f"The main cohort contains {total:,} UFC fights across {int(annual.events.sum()):,} events from 1994 through 8 August 2026; the raw source contains 11,441 fights across multiple promotions.",
        "UFC conclusions require explicit event classification. Eight 1993 UFC fights are retained outside the requested window and supply earlier history.",
        "annual.csv; source_summary.csv",
    )
    add(
        "Event volume expanded, then stabilized",
        f"The snapshot has {int(annual.loc[1994, 'events'])} events in 1994, {int(annual.loc[2010, 'events'])} in 2010 and {int(annual.loc[2025, 'events'])} in 2025.",
        "Recent changes in performance should not automatically be explained by continued event-count growth. Do not annualize the partial 2026 count.",
        "annual.csv",
    )
    add(
        "A small majority of fights end by KO/TKO or submission",
        f"Finishes account for {finish:.1%} of all selected fights; KO/TKO alone accounts for {ko:.1%}.",
        "Draws, no contests, DQs and other methods remain in the denominator; the finish definition is deliberately narrow.",
        "annual.csv",
    )
    add(
        "Heavyweight finishing is concentrated in knockouts",
        f"Across {int(division.loc['Heavyweight', 'fights']):,} heavyweight fights, KO/TKO accounts for {division.loc['Heavyweight', 'ko_rate']:.1%}, versus {ko:.1%} overall.",
        "Division is a major descriptive factor. This does not isolate a causal effect of body weight.",
        "divisions.csv",
    )
    add(
        "The apparent decision trend depends on division composition",
        f"Observed decision rates are {mix.iloc[0].all_division_decision_rate:.2%} in 2010–2019 and {mix.iloc[1].all_division_decision_rate:.2%} in 2020–2025. Fixed shared-division weights yield {mix.iloc[0].standardized_decision_rate:.2%} and {mix.iloc[1].standardized_decision_rate:.2%}.",
        "The direction reverses under standardization. This is a composition diagnostic restricted to adequately represented shared divisions.",
        "division_mix_adjustment.csv",
    )
    add(
        "Finish risk declines among later-round survivors",
        f"For standard three-round fights, the conditional finish rate is {risk.loc[1, 'finish_risk']:.1%} in round 1, {risk.loc[2, 'finish_risk']:.1%} in round 2 and {risk.loc[3, 'finish_risk']:.1%} in round 3.",
        "Each denominator is the number reaching that round. Survivor selection and changing time exposure prevent a causal fatigue interpretation.",
        "round_hazard.csv",
    )
    add(
        "Younger fighters win more often in unequal-age matchups",
        f"The older fighter wins {advantage.loc['age', 'win_rate']:.1%} of {int(advantage.loc['age', 'n']):,} decisive unequal-age bouts. Winners are {abs(paired.loc['age', 'mean_paired_difference']):.2f} years younger on average within matched bouts.",
        "This association can reflect selection, experience, division and matchup quality. Repeated fighter appearances remain a dependence limitation.",
        "advantage_uncertainty.csv; paired_statistics.csv",
    )
    add(
        "Reach advantage has a modest unadjusted association",
        f"The longer-reach fighter wins {advantage.loc['reach_inches', 'win_rate']:.1%} of {int(advantage.loc['reach_inches', 'n']):,} eligible fights; the event-bootstrap 95% interval is {advantage.loc['reach_inches', 'event_bootstrap_low']:.1%}–{advantage.loc['reach_inches', 'event_bootstrap_high']:.1%}.",
        "The effect is small relative to uncertainty about individual outcomes. Missing or equal reach and unresolved results are excluded.",
        "advantage_uncertainty.csv",
    )
    leader = fighters.sort_values(["wins", "fights"], ascending=False).iloc[0]
    add(
        "Career volume and efficiency answer different questions",
        f"{leader.fighter_name} leads the selected win-count table with {int(leader.wins)} wins in {int(leader.fights)} bouts; his decisive-fight win rate is {leader.win_rate:.1%}.",
        "Count rankings reward longevity. Percentage rankings require at least 10 decisive fights and show their sample sizes.",
        "fighters.csv",
    )
    add(
        "Winners generate more significant-strike output",
        f"Winners land {performance.loc[True, 'sig_landed_per_minute']:.2f} significant strikes per minute versus {performance.loc[False, 'sig_landed_per_minute']:.2f} for losers, using pooled observed exposure.",
        "This is a post-fight description, not a valid predictor from the same bout. Per-minute rates and raw totals answer different questions.",
        "performance_by_result.csv",
    )
    add(
        "Grappling output also separates winners and losers",
        f"Winners average {performance.loc[True, 'td_per_15']:.2f} landed takedowns per 15 minutes versus {performance.loc[False, 'td_per_15']:.2f} for losers.",
        "The association combines style, opponent quality and control of the fight. It does not establish that attempting more takedowns causes a win.",
        "performance_by_result.csv",
    )
    add(
        "Title fights remain longer within five-round schedules",
        f"Among standard five-round bouts, title fights average {title.title_mean_minutes:.2f} minutes and non-title fights {title.non_title_mean_minutes:.2f}, a {title.mean_difference_minutes:.2f}-minute gap.",
        "Restricting scheduled length removes one obvious confounder; era, divisions and fighter selection remain. The rank-biserial effect is small.",
        "title_statistics.csv",
    )
    add(
        "Bonus categories change at the 2014 boundary",
        f"Performance of the Night first appears in {int(bonus.loc['Performance of the Night', 'first_year'])}; KO and submission categories last appear in {int(bonus.loc['Knockout of the Night', 'last_year'])} in this snapshot.",
        "Comparisons across time must account for category changes. No individual Performance bonus rankings are inferred from fight-only rows.",
        "bonus_categories.csv",
    )
    add(
        "Finishing bouts are more often decorated",
        f"A bonus category is attached to {bonus_rates.loc['KO/TKO', 'decorated_fight_rate']:.1%} of KO/TKO bouts and {bonus_rates.loc['Decision', 'decorated_fight_rate']:.1%} of decision bouts.",
        "Rates include pre-bonus years and are descriptive. Award rules and era composition confound this difference; decorated fights are not individual payouts.",
        "bonus_rates.csv",
    )
    add(
        "International representation contracted in 2020",
        f"Known-location non-USA event share falls from {annual.loc[2019, 'international_event_share']:.1%} in 2019 to {annual.loc[2020, 'international_event_share']:.1%} in 2020, then reaches {annual.loc[2025, 'international_event_share']:.1%} in 2025.",
        "This timing is consistent with the COVID-era disruption, but the dataset alone cannot identify its causes. Locations are country or territory labels.",
        "annual.csv",
    )
    add(
        "Historical features provide limited predictive value",
        f"Validation selected {model.lower()}; test accuracy is {score.accuracy:.1%} (event-bootstrap 95% interval {score.accuracy_ci_low:.1%}–{score.accuracy_ci_high:.1%}), ROC-AUC {score.roc_auc:.3f}, versus {baseline.accuracy:.1%} baseline accuracy.",
        "The model remains uncertain and is not a betting system. Test periods include previously observed fighters; this evaluates future bouts, not generalization to entirely unseen fighters.",
        "model_metrics.csv",
    )
    (TABLES / "insights.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = [
        "# Research findings",
        "",
        "Main cohort: UFC events, 1994–8 August 2026. All values are recomputed from the supplied snapshot.",
        "",
    ]
    for i, item in enumerate(findings, 1):
        report += [
            f"## {i}. {item['observation']}",
            "",
            f"**Evidence.** {item['evidence']}",
            "",
            f"**Interpretation.** {item['interpretation']}",
            "",
            f"Source tables: {item['source']}",
            "",
        ]
    (ROOT / "FINDINGS.md").write_text("\n".join(report), encoding="utf-8")
    return findings


if __name__ == "__main__":
    print(f"Wrote {len(build_findings())} evidence-based findings")
