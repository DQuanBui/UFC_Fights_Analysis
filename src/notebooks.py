"""Build and execute the eight reproducible research notebooks."""

import argparse
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

from .data_loader import ROOT

SETUP = """
from pathlib import Path
import sys
import io
import pandas as pd
import numpy as np
from IPython.display import display, Image
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'data' / 'raw').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data_loader import load_raw, profile_sources, verify_raw, TABLES
from src.pipeline import prepare
from src.analysis import *
from src.visualization import *
def show(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format='png', bbox_inches='tight')
    display(Image(data=buffer.getvalue()))
    plt.close(figure)
"""
PREP = """
data = prepare(save=False)
fights, appearances = cohort(data)
print(f"Cohort: {len(fights):,} fights, {fights.event_id.nunique():,} events, {appearances.fighter_id.nunique():,} fighters")
print(f"Date range: {fights.event_date.min():%Y-%m-%d} to {fights.event_date.max():%Y-%m-%d}")
"""


def build_notebooks():
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    books = {}

    def new(name, title, introduction):
        cells = [
            nbf.v4.new_markdown_cell(f"# {title}\n\n{introduction}"),
            nbf.v4.new_code_cell(textwrap.dedent(SETUP).strip()),
        ]
        books[name] = cells
        return cells

    def section(cells, title, description, code):
        cells.extend(
            [
                nbf.v4.new_markdown_cell(f"## {title}\n\n{description}"),
                nbf.v4.new_code_cell(textwrap.dedent(code).strip()),
            ]
        )

    cells = new(
        "01_data_understanding",
        "01 · Understand the source snapshot",
        "Research question: what does each row represent, and which questions can this snapshot answer? Inspect all seven sources before adopting a fact table. Raw hashes protect the supplied files.",
    )
    section(
        cells,
        "Load and verify",
        "All sources are local. No Kaggle credentials or network download are needed.",
        """
        print(f"Verified {verify_raw()} original source hashes")
        raw = load_raw()
        profiles = profile_sources(raw)
        display(profiles['source_summary'])
        display(profiles['date_coverage'])
        """,
    )
    for name in [
        "event",
        "fight",
        "fighter",
        "fighter_bonus",
        "master",
        "round",
        "scrape_error",
    ]:
        section(
            cells,
            f"Inspect {name}.csv",
            "Inspect sample rows and every column’s inferred type, missingness, uniqueness and example values.",
            f"""
        display(raw['{name}'].head(3))
        with pd.option_context('display.max_rows', 110, 'display.max_colwidth', 70):
            display(profiles['data_dictionary'].query("table == '{name}'"))
        """,
        )
    section(
        cells,
        "Relationships and categorical coverage",
        "event → fight is one-to-many; fighter joins twice to fight; round has a fight + round key; bonus has a fight + category key. Names are not keys.",
        """
        for source, column, target, key in [('fight','event_id','event','event_id'),
                                           ('fight','r_id','fighter','fighter_id'),
                                           ('fight','b_id','fighter','fighter_id'),
                                           ('round','fight_id','fight','fight_id'),
                                           ('fighter_bonus','fight_id','fight','fight_id')]:
            print(source, column, 'orphan references:', (~raw[source][column].isin(raw[target][key])).sum())
        display(profiles['category_counts'].query("column in ['result_status','method','stance','bonus_type','error_name']"))
        """,
    )
    section(
        cells,
        "Why master is insufficient",
        "The wide master has the same fight IDs, joined attributes and aggregate statistics. It loses round trajectories, omits fighters without a fight and zero-fills missing round totals. Lifetime profile metrics also contain future information for historical fights.",
        """
        missing_round_ids = set(raw['fight'].fight_id) - set(raw['round'].fight_id)
        affected = raw['master'][raw['master'].fight_id.isin(missing_round_ids)]
        display(affected[['fight_id','event_date','r_total_sig_landed','b_total_sig_landed','rounds_fought']].head(10))
        print('Fights with no round records:', len(affected))
        print('Logged fight errors:', raw['scrape_error'].entity_id.nunique())
        print('Profiles without observed fights:', (~raw['fighter'].fighter_id.isin(set(raw['fight'].r_id) | set(raw['fight'].b_id))).sum())
        """,
    )

    cells = new(
        "02_data_cleaning",
        "02 · Clean, validate and engineer features",
        "Preserve raw observations, validate join cardinality, distinguish missing statistics from zeros, and calculate fighter history using strictly earlier dates. See DATA_NOTES.md for every important decision.",
    )
    section(
        cells,
        "Build analytical tables",
        "No raw source is overwritten. Output includes all raw fights with explicit scope flags.",
        PREP,
    )
    section(
        cells,
        "Data quality ledger",
        "Zero-count checks remain visible in the exported ledger. Nonzero issues document actions rather than disappearing through row deletion.",
        """
        display(data['quality'].query('affected_rows > 0'))
        print('All fight rows preserved:', len(data['fights']) == len(load_raw()['fight']))
        print('Unique fact IDs:', data['fights'].fight_id.is_unique)
        print('Two appearances per fight:', len(data['appearances']) == 2*len(data['fights']))
        """,
    )
    section(
        cells,
        "Scope and coverage",
        "UFC main cards and TUF finales are distinguished from other promotions and feeder series. Coverage starts at 1994 for reporting; 1993 UFC bouts remain available to earlier-history features.",
        """
        display(data['fights'].groupby(['is_ufc','in_scope']).agg(fights=('fight_id','size'), events=('event_id','nunique')))
        coverage = data['fights'].groupby(['fight_year','is_ufc']).round_stats_complete.agg(['count','mean'])
        display(coverage.head(20))
        """,
    )
    section(
        cells,
        "Duration rules",
        "Early formats include 10-minute rounds, overtime and no-time-limit bouts. A universal five-minute formula would be wrong.",
        """
        from src.cleaning import fight_duration
        examples = [('3 Rnd (10-5-5)', 2, '2:30'), ('1 Rnd + 2OT (15-3-3)', 3, '2:00'), ('No Time Limit', 1, '25:00')]
        display(pd.DataFrame([dict(format=f, ending_round=r, clock=t, seconds=fight_duration(f,r,t)) for f,r,t in examples]))
        display(fights.groupby('time_format').fight_duration_seconds.agg(['count','mean','max']))
        """,
    )
    section(
        cells,
        "Inspect the feature timeline",
        "Each current fight is excluded. Every fight on the same date shares the same earlier-date history because tournament order is unavailable.",
        """
        sample_id = appearances.fighter_id.value_counts().index[0]
        timeline = data['appearances'].query('fighter_id == @sample_id and is_ufc').sort_values('event_date')
        display(timeline[['event_date','fighter_name','won','prior_ufc_fights','prior_wins','prior_win_streak','prior_sig_per_minute']].tail(12))
        assert not appearances.duplicated(['fight_id','fighter_id']).any()
        assert appearances.loc[~appearances.round_stats_complete,'sig_landed'].isna().all()
        """,
    )

    cells = new(
        "03_eda_ufc_history",
        "03 · UFC growth, geography and composition",
        "Research question: how did event volume, participation and geographic representation change? Counts describe the supplied cohort, not a independently certified event census. The final year is incomplete.",
    )
    section(
        cells,
        "Prepare cohort",
        "Restrict headlines to 1994–2026 UFC event scope.",
        PREP,
    )
    section(
        cells,
        "Scale and participation",
        "Compare full calendar years; the hollow marker denotes the incomplete final year.",
        """
        annual = annual_summary(fights, appearances)
        display(annual)
        show(history_chart(fights, appearances))
        print('Events in 1994 / 2010 / 2025:', annual.set_index('fight_year').loc[[1994,2010,2025],'events'].to_dict())
        """,
    )
    section(
        cells,
        "International representation",
        "Country is parsed from the final location segment. Source territory labels remain unchanged; missing locations are excluded from geographic rates.",
        """
        events = event_summary(fights)
        display(events.groupby('country',dropna=False).agg(events=('event_id','size'), first_year=('event_year','min')).sort_values('events',ascending=False))
        display(annual.query('2018 <= fight_year <= 2025')[['fight_year','events','international_event_share','countries_or_territories']])
        """,
    )
    section(
        cells,
        "Cards and representation",
        "Title indicators are observed; main-event status and venue names are unavailable. Women’s participation is inferred from division labels.",
        """
        display(events.groupby('event_category').agg(events=('event_id','size'), mean_fights=('fights','mean'), mean_finish_rate=('finish_rate','mean')))
        display(annual[['fight_year','women_bout_share','divisions','title_rate','fights_per_event']].tail(15))
        display(events.nlargest(10,'fights')[['event_name','event_date','fights','finish_rate']])
        """,
    )
    section(
        cells,
        "Era comparison",
        "Calendar decades are transparent descriptive bins, not claims about exact rule or ownership changes. The first and last bins have incomplete calendar coverage.",
        """
        era_tables = export_outcomes(data)
        display(era_tables['era_comparison'])
        show(outcome_chart(fights))
        """,
    )

    cells = new(
        "04_fighter_analysis",
        "04 · Fighters, attributes and divisions",
        "Research question: which fighters stand out in volume and efficiency, and how do attributes vary across divisions? Name collisions remain separate by fighter ID. Profile traits are snapshots.",
    )
    section(
        cells,
        "Prepare cohort",
        "All rates and records below use the selected UFC window.",
        PREP,
    )
    section(
        cells,
        "Volume versus efficiency",
        "Win rates divide wins by decisive fights. Finish-win rates divide finish wins by all bouts. Require 10 decisive fights for percentage leaderboards.",
        """
        leaders = fighter_summary(appearances)
        display(leaders[['fighter_name','fighter_id','fights','wins','losses','other_results','ko_wins','submission_wins','longest_win_streak_conservative']].head(20))
        display(leaders.query('decisive_fights >= 10').nlargest(15,'win_rate')[['fighter_name','decisive_fights','wins','win_rate','finish_win_rate']])
        """,
    )
    section(
        cells,
        "Division comparison",
        "Fight-weighted and appearance-weighted averages are labeled separately. Tiny historical categories are omitted from the chart, but remain in the table.",
        """
        divisions = division_summary(fights, appearances)
        display(divisions)
        show(division_chart(fights, appearances))
        """,
    )
    section(
        cells,
        "Age, height and reach",
        "Use one row per fighter for height/reach correlation. Winner/loser ages are fight-level measurements and repeat athletes.",
        """
        show(physical_chart(appearances))
        physical = pd.concat([attribute_advantage(fights, c) for c in ['age','height_inches','reach_inches']])
        display(physical.query('unequal_pairs >= 100'))
        """,
    )
    section(
        cells,
        "Stance and matchup outcomes",
        "Stance is a present-day profile label, not a historical record. These associations may reflect division composition and fighter quality.",
        """
        tables = export_fighters(data)
        display(tables['stance'])
        matchup = fights[fights.decisive & (((fights.r_stance == 'Southpaw') & (fights.b_stance == 'Orthodox')) | ((fights.b_stance == 'Southpaw') & (fights.r_stance == 'Orthodox')))]
        southpaw_id = matchup.r_id.where(matchup.r_stance.eq('Southpaw'), matchup.b_id)
        print('Southpaw vs orthodox bouts:', len(matchup))
        print('Southpaw win share:', matchup.winner_id.eq(southpaw_id).mean())
        display(tables['age_profile'])
        """,
    )

    cells = new(
        "05_fight_analysis",
        "05 · Outcomes, duration and composition",
        "Research question: do changes in aggregate finish rates survive division adjustment? Finishes mean KO/TKO, including doctor stoppages, or submission. Result status overrides method for draws and no contests.",
    )
    section(
        cells,
        "Prepare cohort",
        "Every outcome remains in the denominator of method shares.",
        PREP,
    )
    section(
        cells,
        "Victory methods",
        "KO versus TKO is not separately recoverable from the combined source category. Do not invent that split.",
        """
        tables = export_outcomes(data)
        display(tables['outcome_methods'])
        show(outcome_chart(fights))
        print('Finish share:', fights.is_finish.mean())
        """,
    )
    section(
        cells,
        "A composition reversal",
        "Compare full 2010–2019 and 2020–2025 periods. Standardization uses fixed pooled weights over shared divisions with at least 50 fights in each period.",
        """
        adjustment = composition_adjustment(fights)
        display(adjustment)
        print('Observed change (percentage points):', adjustment.all_division_decision_rate.diff().iloc[-1]*100)
        print('Standardized change (percentage points):', adjustment.standardized_decision_rate.diff().iloc[-1]*100)
        """,
    )
    section(
        cells,
        "Duration and scheduled exposure",
        "Title fights often have longer schedules. Show schedule-stratified duration before comparing title status.",
        """
        display(tables['title_duration'])
        display(fights.groupby(['weight_class','outcome_group']).fight_duration_seconds.agg(['count','mean','median']).head(30))
        display(tables['ending_rounds'])
        """,
    )
    section(
        cells,
        "How long are bouts still active?",
        "This empirical survivor curve counts every ending, including decisions, as an endpoint. It is not a Kaplan–Meier estimate of finish risk with decisions censored.",
        """
        curve = duration_curve(fights)
        style()
        fig, ax = plt.subplots(figsize=(9,5))
        for schedule, rows in curve.groupby('scheduled_rounds'):
            ax.step(rows.seconds/60, rows.still_fighting, where='post', label=f'{int(schedule)} scheduled rounds')
        ax.set(xlabel='Elapsed fight time (minutes)', ylabel='Fraction still fighting')
        ax.legend(); show(finish(fig,'The duration distribution reflects scheduled limits'))
        """,
    )

    cells = new(
        "06_round_analysis",
        "06 · Round trajectories, striking and grappling",
        "Research question: how do finish risk and activity change within fights? The round table supports detailed strike targets, positions, takedowns and control. Missing time is not zero control.",
    )
    section(
        cells,
        "Prepare cohort",
        "Round statistics are linked by fight ID and validated against corner identities.",
        PREP,
    )
    section(
        cells,
        "Risk sets",
        "The denominator is fights reaching each round, separately by schedule. Late-round fighters are a selected subset.",
        """
        display(round_hazard(fights))
        show(hazard_chart(fights))
        """,
    )
    section(
        cells,
        "Activity and coverage",
        "Only observed rounds with positive elapsed time contribute exposure. Fight-level totals require complete round sequences.",
        """
        rounds = data['rounds'][data['rounds'].in_scope].copy()
        tables = export_performance(data)
        display(tables['round_activity'])
        display(rounds[['round_no','sig_landed','sig_atmp','td_success','td_atmp','ctrl_seconds','round_duration_seconds']].describe())
        """,
    )
    section(
        cells,
        "Striking and grappling through time",
        "Rates pool landed actions and exposure. Comparing early and later years requires coverage awareness.",
        """
        show(performance_chart(appearances))
        display(tables['performance_by_year'].tail(12))
        display(tables['performance_by_division'])
        """,
    )
    section(
        cells,
        "Within-fight striking change",
        "Restrict to fighter-bouts with observed rounds 1 and 2. Comparing the same fighters reduces composition differences but does not remove survival selection.",
        """
        rounds['rate'] = rounds.sig_landed / (rounds.round_duration_seconds.where(rounds.round_duration_seconds.gt(0))/60)
        paired = rounds.pivot(index=['fight_id','fighter_id'], columns='round_no', values='rate')[[1,2]].dropna()
        print('Paired fighter-bouts:', len(paired))
        print('Mean change from round 1 to round 2:', (paired[2]-paired[1]).mean())
        display(rounds.groupby('round_no').ctrl_seconds.agg(['count','mean']))
        """,
    )
    section(
        cells,
        "Bonuses and style",
        "Bonus records identify fight/category combinations. Individual Performance recipients and cash amounts cannot be reconstructed.",
        """
        display(tables['bonus_categories'])
        display(tables['bonus_rates'])
        display(tables['fotn_appearances'].head(15))
        """,
    )

    cells = new(
        "07_statistical_analysis",
        "07 · Effect sizes and uncertainty",
        "Research question: which associations are practically meaningful? Primary evidence is effect size and event-clustered bootstrap intervals. All tests are exploratory; repeated fighters across cards remain dependent.",
    )
    section(
        cells,
        "Prepare cohort",
        "Winner and loser measurements are paired within the same fight.",
        PREP,
    )
    section(
        cells,
        "Paired winner–loser comparisons",
        "Null: paired differences are symmetric about zero for Wilcoxon. Signed-rank tests require symmetry and independent pairs; repeated fighters weaken independence. Holm adjustment covers this family of nine tests. Bootstrap intervals resample events.",
        """
        from src.statistics import export_statistics
        results = export_statistics(data)
        display(results['paired_statistics'])
        """,
    )
    section(
        cells,
        "Unequal-attribute matchups",
        "Wilson intervals treat bouts as Bernoulli observations. Event-bootstrap intervals additionally preserve within-card dependence. Neither identifies a causal advantage.",
        """
        display(results['advantage_uncertainty'])
        display(attribute_advantage(fights,'reach_inches').query('unequal_pairs >= 100'))
        """,
    )
    section(
        cells,
        "Division and finishing method",
        "Null: method and division are independent among KO/TKO, submission and decision bouts. Inspect expected cell counts, report Cramér’s V, and acknowledge repeated competitors.",
        """
        display(results['division_association'])
        """,
    )
    section(
        cells,
        "Physical correlation",
        "Null: no linear/monotonic association between height and reach. Use one row per fighter; pooling divisions can drive correlations.",
        """
        display(results['physical_correlation'])
        """,
    )
    section(
        cells,
        "Title fights at a common schedule",
        "Compare only standard five-round fights. Mann–Whitney tests distributions, not necessarily means. A small p-value can accompany a small rank-biserial effect; title assignment is not randomized.",
        """
        display(results['title_statistics'])
        """,
    )

    cells = new(
        "08_machine_learning",
        "08 · Predicting outcomes from prior history",
        "Research question: can information available before a fight add predictive value? This is sequential historical replay, not a betting system. A/B assignment is a deterministic outcome-independent hash of fight ID.",
    )
    section(
        cells,
        "Prepare chronological features",
        "Each fighter history excludes the entire current date. No lifetime profile performance metrics or current-bout statistics enter the allowlist.",
        PREP,
    )
    section(
        cells,
        "Features and time splits",
        "Train through 2019; validation 2020–2022 selects the model by log loss; test begins in 2023. Later test bouts may use earlier test results in history, as would be known at their dates. Model parameters stay fixed.",
        """
        from src.modeling import model_frame, train_models, FEATURES
        model_data = model_frame(data['fights'])
        print('Allowed features:', FEATURES)
        display(model_data.groupby('split',sort=False).agg(fights=('fight_id','size'), first_date=('event_date','min'), last_date=('event_date','max')))
        display(model_data[FEATURES].isna().mean().rename('Missing share'))
        """,
    )
    section(
        cells,
        "Train interpretable candidates",
        "Fit imputers, scaling and one-hot encoding only within training pipelines. Compare a majority baseline, logistic regression, a shallow tree, random forest and gradient boosting. No hyperparameters are selected on test results.",
        """
        results = train_models(data['fights'])
        display(results['model_metrics'])
        import json
        metadata = json.loads((TABLES/'model_metadata.json').read_text(encoding='utf-8'))
        print('Validation-selected model:', metadata['selected_model'])
        """,
    )
    section(
        cells,
        "Discrimination and calibration",
        "Accuracy at the fixed 0.5 threshold and ROC-AUC measure different properties. Calibration and Brier score assess probability quality.",
        """
        display(results['model_confusion'])
        display(results['model_calibration'])
        roc=results['roc_curves']; style()
        fig,ax=plt.subplots(figsize=(9,5))
        for name,rows in roc.groupby('model'):
            ax.plot(rows.false_positive_rate,rows.true_positive_rate,label=name)
        ax.plot([0,1],[0,1],'--',color='gray');ax.legend()
        ax.set(xlabel='False positive rate',ylabel='True positive rate')
        show(finish(fig,'Held-out ROC curves'))
        """,
    )
    section(
        cells,
        "Interpretation and failure modes",
        "Logistic coefficients are standardized conditional associations. Correlated experience features complicate isolated interpretation. Report subgroup sample sizes and avoid ranking tiny groups.",
        """
        display(results['model_coefficients'].sort_values('coefficient',key=abs,ascending=False).head(15))
        display(results['model_subgroups'])
        """,
    )
    cells.append(
        nbf.v4.new_markdown_cell(
            "## Limits of historical replay\n\nThe snapshot has no timestamped corrections, injury reports, odds or historical rankings. Date of birth is assumed stable; profile stance, weight, reach and career averages are excluded from modeling. Previously seen athletes can appear in later periods, so this is not an unseen-fighter evaluation. Test performance must not guide further model selection without a new holdout."
        )
    )
    paths = []
    from .research_notebooks import cells_for

    for name, cells in books.items():
        cells.extend(cells_for(name[:2]))
        notebook = nbf.v4.new_notebook(
            cells=cells,
            metadata={
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.13"},
            },
        )
        path = folder / f"{name}.ipynb"
        nbf.write(notebook, path)
        paths.append(path)
    return paths


def execute_notebooks(paths=None):
    paths = paths or sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        notebook = nbf.read(path, as_version=4)
        NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()
        nbf.write(notebook, path)
        print(
            f"Executed {path.name}: {sum(c.cell_type == 'code' for c in notebook.cells)} code cells",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.build:
        build_notebooks()
    if args.execute:
        execute_notebooks()
