"""Interactive UFC research explorer. Run: streamlit run dashboard/app.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.analysis import (
    annual_summary,
    composition_adjustment,
    division_summary,
    event_summary,
    fighter_summary,
    performance_summary,
    round_hazard,
)
from src.pipeline import prepare
from src.visualization import COLORS, plotly_style

st.set_page_config(page_title="UFC | Fight Analytics", page_icon="🥊", layout="wide")


@st.cache_data(show_spinner="Preparing the local fight snapshot…")
def load_data():
    return prepare(save=False)


@st.cache_data
def rankings(appearances):
    return fighter_summary(appearances)


@st.cache_data
def load_table(name):
    return pd.read_csv(ROOT / "outputs" / "tables" / f"{name}.csv")


def chart(figure):
    st.plotly_chart(plotly_style(figure), width="stretch")


def table(frame):
    st.dataframe(frame, hide_index=True, width="stretch")


def download(frame, label="Download filtered fights"):
    st.download_button(
        label, frame.to_csv(index=False).encode("utf-8"), "ufc_filtered.csv", "text/csv"
    )


def line(frame, x, y, title, labels=None, color=None):
    if x == "fight_year" and "partial_year" in frame:
        figure = px.line(
            frame[~frame.partial_year],
            x=x,
            y=y,
            color=color,
            markers=True,
            title=title,
            labels=labels,
        )
        partial = frame[frame.partial_year]
        if not partial.empty:
            point = px.scatter(partial, x=x, y=y).data[0]
            point.update(
                name="2026 (partial)",
                showlegend=True,
                marker=dict(symbol="circle-open", size=11, color=COLORS[0]),
            )
            figure.add_trace(point)
    else:
        figure = px.line(
            frame, x=x, y=y, color=color, markers=True, title=title, labels=labels
        )
    if any(term in y for term in ["share", "rate", "coverage"]):
        figure.update_yaxes(tickformat=".0%")
    chart(figure)


data = load_data()
all_fights = data["fights"][data["fights"].in_scope].copy()
all_appearances = data["appearances"][data["appearances"].in_scope].copy()
names = data["fighters"].set_index("fighter_id").fighter_name.to_dict()


def label(fid):
    return f"{names.get(fid, fid)} · {fid[-6:]}"


with st.sidebar:
    st.markdown("### UFC / ANALYTICS")
    st.caption("EVOLUTION • PERFORMANCE • OUTCOMES")
    page = st.radio(
        "Explore",
        [
            "Overview",
            "UFC Evolution",
            "Fighters",
            "Fight Outcomes",
            "Weight Classes",
            "Striking & Grappling",
            "Events",
            "Bonuses",
            "Fighter Comparison",
            "Fight Prediction / ML",
            "Key Insights",
        ],
    )
    st.divider()
    st.caption(
        "Local snapshot: 1994–2026\n\nLatest fight: "
        + all_fights.event_date.max().strftime("%d %b %Y")
    )

fights = all_fights.copy()
fixed_page = page in ["Fight Prediction / ML", "Key Insights"]
if not fixed_page:
    with st.sidebar:
        years = st.slider("Fight years", 1994, 2026, (1994, 2026))
        divisions = st.multiselect(
            "Weight classes", sorted(all_fights.weight_class.unique())
        )
        outcomes = st.multiselect("Outcomes", sorted(all_fights.outcome_group.unique()))
        with st.expander("More filters"):
            gender = st.multiselect("Gender label", sorted(all_fights.gender.unique()))
            countries = st.multiselect(
                "Countries / territories", sorted(all_fights.country.dropna().unique())
            )
            fighter_ids = st.multiselect(
                "Fighters",
                sorted(all_appearances.fighter_id.unique(), key=label),
                format_func=label,
            )
            event_ids = st.multiselect(
                "Events",
                all_fights.sort_values("event_date", ascending=False)
                .event_id.drop_duplicates()
                .tolist(),
                format_func=lambda eid: all_fights.loc[
                    all_fights.event_id.eq(eid), "event_name"
                ].iloc[0],
            )
    fights = fights[fights.fight_year.between(*years)]
    for column, selected in [
        ("weight_class", divisions),
        ("outcome_group", outcomes),
        ("gender", gender),
        ("country", countries),
        ("event_id", event_ids),
    ]:
        if selected:
            fights = fights[fights[column].isin(selected)]
    if fighter_ids:
        fights = fights[fights.r_id.isin(fighter_ids) | fights.b_id.isin(fighter_ids)]
appearances = all_appearances[all_appearances.fight_id.isin(fights.fight_id)]

st.markdown(
    '<div style="font-size:13px;letter-spacing:3px;color:#C8383D;font-weight:700">THE FIGHT, IN NUMBERS</div>',
    unsafe_allow_html=True,
)
st.title(page if page != "Overview" else "Three decades inside the Octagon")
if fixed_page:
    st.caption("Full research cohort • fixed evaluation periods • 2026 is partial")
else:
    st.caption(
        f"{len(fights):,} fights · {fights.event_id.nunique():,} events · {appearances.fighter_id.nunique():,} fighters in selection · 2026 is partial"
    )
st.caption(
    "UFC cards and TUF finales; other promotions and feeder series excluded. All results describe the supplied snapshot."
)

if fights.empty:
    st.info("No fights match this combination. Clear a filter to continue.")
    st.stop()

if page == "Overview":
    st.write(
        "Trace UFC growth, compare fighting styles, and examine what pre-fight history can—and cannot—explain about winning."
    )
    metrics = [
        ("Fights", f"{len(fights):,}"),
        ("Events", f"{fights.event_id.nunique():,}"),
        ("Fighters", f"{appearances.fighter_id.nunique():,}"),
        ("Countries / territories", str(fights.country.nunique())),
    ]
    for column, (title, value) in zip(st.columns(4), metrics):
        column.metric(title, value)
    for column, (title, value) in zip(
        st.columns(4),
        [
            ("KO / TKO", f"{fights.is_ko.mean():.1%}"),
            ("Submissions", f"{fights.is_submission.mean():.1%}"),
            ("Decisions", f"{fights.is_decision.mean():.1%}"),
            (
                "Average duration",
                f"{fights.fight_duration_seconds.mean() / 60:.1f} min",
            ),
        ],
    ):
        column.metric(title, value)
    st.caption(
        "Outcome percentages use all selected fights. Finishes mean KO/TKO or submission; draws, no contests and other methods remain in the denominator."
    )
    left, right = st.columns([1.25, 1])
    annual = annual_summary(fights, appearances)
    with left:
        line(
            annual,
            "fight_year",
            "fights",
            "Annual fight volume",
            {"fight_year": "Year", "fights": "Fights"},
        )
    with right:
        mix = (
            fights.outcome_group.value_counts()
            .rename_axis("Outcome")
            .reset_index(name="Fights")
        )
        chart(
            px.bar(
                mix,
                x="Fights",
                y="Outcome",
                orientation="h",
                color="Outcome",
                title="How the selected fights ended",
                color_discrete_sequence=COLORS,
            )
        )
    st.info(
        f"{fights.round_stats_complete.mean():.1%} of selected fights have complete round statistics. Early missing statistics are excluded from performance rates."
    )
    download(
        fights[
            [
                "event_date",
                "event_name",
                "r_fighter_name",
                "b_fighter_name",
                "weight_class",
                "outcome_group",
                "fight_duration_seconds",
            ]
        ]
    )

elif page == "UFC Evolution":
    annual = annual_summary(fights, appearances)
    metric = st.selectbox(
        "Trend",
        [
            "events",
            "fights",
            "active_fighters",
            "women_bout_share",
            "international_event_share",
            "fights_per_event",
            "title_rate",
            "stats_coverage",
        ],
    )
    line(
        annual,
        "fight_year",
        metric,
        metric.replace("_", " ").title(),
        {"fight_year": "Year", metric: metric.replace("_", " ")},
    )
    st.caption(
        "Growth for 2026 is withheld because the snapshot stops in August. International shares use events with known locations; Puerto Rico is retained as a source territory label."
    )
    table(annual)
    if fights.fight_year.lt(2020).any() and fights.fight_year.between(2020, 2025).any():
        st.subheader("Does the division mix explain the decision trend?")
        adjustment = composition_adjustment(fights)
        table(adjustment)
        st.caption(
            "Full 2010–2019 vs 2020–2025 years. Shared divisions require 50 fights per period; fixed pooled division weights isolate a composition comparison, not a causal effect."
        )

elif page == "Fighters":
    summaries = rankings(appearances)
    minimum = st.slider("Minimum decisive fights for percentage rankings", 5, 30, 10)
    rank_metric = st.selectbox(
        "Rank by",
        [
            "wins",
            "fights",
            "ko_wins",
            "submission_wins",
            "finish_wins",
            "win_rate",
            "finish_win_rate",
            "title_fights",
            "observed_career_years",
            "longest_win_streak_conservative",
        ],
    )
    ranked = (
        summaries[summaries.decisive_fights.ge(minimum)]
        if "rate" in rank_metric
        else summaries
    )
    table(
        ranked.sort_values(rank_metric, ascending=False)[
            [
                "fighter_name",
                "fighter_id",
                "fights",
                "wins",
                "losses",
                "other_results",
                rank_metric,
            ]
        ]
        .loc[:, lambda x: ~x.columns.duplicated()]
        .head(25)
    )
    selected = st.selectbox(
        "Fighter profile", sorted(summaries.fighter_id, key=label), format_func=label
    )
    profile = summaries[summaries.fighter_id.eq(selected)].iloc[0]
    st.subheader(names[selected])
    for column, (title, value) in zip(
        st.columns(4),
        [
            ("Record in selection", f"{profile.wins}–{profile.losses}"),
            ("Other results", profile.other_results),
            ("Finish wins", profile.finish_wins),
            ("Title bouts", profile.title_fights),
        ],
    ):
        column.metric(title, value)
    table(
        pd.DataFrame(
            [
                profile[
                    [
                        "height_inches",
                        "reach_inches",
                        "profile_weight_lbs",
                        "snapshot_stance",
                        "first_fight",
                        "last_fight",
                        "stats_fights",
                    ]
                ]
            ]
        )
    )
    career = (
        appearances[appearances.fighter_id.eq(selected)]
        .sort_values("event_date")
        .copy()
    )
    career["opponent"] = career.opponent_id.map(names)
    career["result"] = np.select(
        [career.won, career.lost], ["Win", "Loss"], default="Draw / no contest"
    )
    chart(
        px.scatter(
            career,
            x="event_date",
            y="sig_per_minute",
            color="result",
            hover_data=["opponent", "event_name"],
            title="Significant strikes per minute by fight",
            labels={"event_date": "Fight date", "sig_per_minute": "Landed / minute"},
        )
    )
    table(
        career[
            [
                "event_date",
                "event_name",
                "opponent",
                "result",
                "outcome_group",
                "age",
                "sig_per_minute",
                "td_per_15",
            ]
        ]
    )
    st.caption(
        "Records and streaks use the selected cohort. Career spans are observed spans, not retirement estimates. Mixed-result same-day tournament dates reset the conservative streak."
    )

elif page == "Fight Outcomes":
    rates = (
        pd.crosstab(fights.fight_year, fights.outcome_group, normalize="index")
        .mul(100)
        .reset_index()
        .melt("fight_year", var_name="Outcome", value_name="Share (%)")
    )
    chart(
        px.bar(
            rates,
            x="fight_year",
            y="Share (%)",
            color="Outcome",
            title="Outcome mix by year",
            labels={"fight_year": "Year"},
        )
    )
    hazard = round_hazard(fights)
    if not hazard.empty:
        hazard["Schedule"] = hazard.scheduled_rounds.astype(str) + " rounds"
        chart(
            px.line(
                hazard,
                x="round_no",
                y="finish_risk",
                color="Schedule",
                markers=True,
                hover_data=["at_risk", "finishes"],
                title="Finish risk among fights reaching each round",
                labels={
                    "round_no": "Round",
                    "finish_risk": "Conditional finish probability",
                },
            )
        )
        table(hazard)
    chart(
        px.box(
            fights,
            x="outcome_group",
            y="fight_duration_seconds",
            color="outcome_group",
            title="Duration by result",
            labels={
                "outcome_group": "Result",
                "fight_duration_seconds": "Duration (seconds)",
            },
        )
    )
    st.caption(
        "Risk sets separate three- and five-round standard schedules. Later rounds contain survivors of earlier rounds; this is a descriptive comparison."
    )

elif page == "Weight Classes":
    divisions = division_summary(fights, appearances)
    minimum = st.slider("Minimum fights per division", 1, 300, 100)
    eligible = divisions[divisions.fights.ge(minimum)]
    chart(
        px.scatter(
            eligible,
            x="mean_sig_per_minute",
            y="mean_td_per_15",
            size="fights",
            color="finish_rate",
            hover_name="weight_class",
            title="Division styles: striking, takedowns and finishes",
            labels={
                "mean_sig_per_minute": "Mean fight-level strikes / minute",
                "mean_td_per_15": "Mean takedowns / 15 minutes",
                "finish_rate": "Finish share",
            },
            color_continuous_scale="Tealrose",
        )
    )
    table(eligible)
    st.caption(
        "Physical averages are appearance-weighted profile measurements. Fight-level average rates differ from pooled exposure rates on Striking & Grappling."
    )

elif page == "Striking & Grappling":
    grouping = st.selectbox(
        "Compare by", ["weight_class", "fight_year", "outcome_group"]
    )
    performance = performance_summary(appearances, grouping)
    metric = st.selectbox(
        "Performance metric",
        [
            "sig_landed_per_minute",
            "sig_accuracy",
            "td_per_15",
            "td_accuracy",
            "knockdowns_per_15",
            "sub_attempts_per_15",
            "head_share",
            "body_share",
            "leg_share",
            "distance_share",
            "clinch_share",
            "ground_share",
        ],
    )
    chart(
        px.bar(
            performance,
            x=grouping,
            y=metric,
            title=metric.replace("_", " ").title(),
            labels={
                grouping: grouping.replace("_", " "),
                metric: metric.replace("_", " "),
            },
        )
    )
    table(performance)
    comparison = performance_summary(appearances[appearances.decisive], "won")
    st.subheader("Winner and loser performance")
    table(comparison)
    st.caption(
        "These statistics occur during the fight and help describe outcomes. They are excluded from that fight’s prediction features. Exposure rates pool only observed, complete fights with positive duration."
    )

elif page == "Events":
    events = event_summary(fights).sort_values("event_date", ascending=False)
    geo = (
        events.groupby("country", dropna=False)
        .size()
        .rename("Events")
        .reset_index()
        .sort_values("Events", ascending=False)
    )
    chart(
        px.bar(
            geo.head(15),
            x="country",
            y="Events",
            title="Host countries and territories",
            labels={"country": "Source location label"},
        )
    )
    event = st.selectbox(
        "Event",
        events.event_id.tolist(),
        format_func=lambda eid: events.set_index("event_id").loc[eid, "event_name"],
    )
    row = events[events.event_id.eq(event)].iloc[0]
    st.write(f"**{row.event_name}** · {row.event_date:%d %b %Y} · {row.location}")
    for column, (title, value) in zip(
        st.columns(3),
        [
            ("Selected bouts", row.fights),
            ("Finish share", f"{row.finish_rate:.1%}"),
            ("Title bouts", row.title_fights),
        ],
    ):
        column.metric(title, value)
    bouts = fights[fights.event_id.eq(event)].copy()
    bouts["winner"] = bouts.winner_id.map(names)
    table(
        bouts[
            [
                "r_fighter_name",
                "b_fighter_name",
                "winner",
                "result_status",
                "weight_class",
                "method",
                "finish_round",
                "finish_time",
            ]
        ]
    )
    st.caption(
        "Event summaries reflect the active filters and may contain only part of a card. Venue names and reliable card order are absent from the source."
    )

elif page == "Bonuses":
    bonuses = data["bonuses"].merge(
        fights[["fight_id", "fight_year", "weight_class", "outcome_group"]],
        on="fight_id",
        validate="many_to_one",
    )
    st.info(
        "The source stores fight + bonus category, without recipient IDs or payment amounts. Counts below are decorated fights or fight-category records, not individual awards."
    )
    by_year = (
        bonuses.groupby(["fight_year", "bonus_type"])
        .size()
        .rename("Records")
        .reset_index()
    )
    chart(
        px.bar(
            by_year,
            x="fight_year",
            y="Records",
            color="bonus_type",
            title="Bonus categories across years",
            labels={"fight_year": "Year", "bonus_type": "Category"},
        )
    )
    decorated = (
        fights.assign(decorated=fights.fight_id.isin(bonuses.fight_id))
        .groupby("outcome_group")
        .decorated.agg(["size", "sum", "mean"])
        .reset_index()
    )
    decorated.columns = ["Result", "Fights", "Decorated fights", "Decorated share"]
    table(decorated)
    fotn = bonuses[bonuses.bonus_type.eq("Fight of the Night")].fight_id
    leaders = (
        appearances[appearances.fight_id.isin(fotn)]
        .groupby(["fighter_id", "fighter_name"])
        .size()
        .rename("FOTN bout appearances")
        .reset_index()
        .sort_values("FOTN bout appearances", ascending=False)
    )
    st.subheader("Appearances in Fight of the Night bouts")
    table(leaders.head(20))
    st.caption(
        "Performance, KO and submission recipient rankings are withheld because recipient identity is not recorded."
    )

elif page == "Fighter Comparison":
    ids = sorted(appearances.fighter_id.unique(), key=label)
    left, right = st.columns(2)
    with left:
        a = st.selectbox("Fighter A", ids, format_func=label)
    with right:
        b = st.selectbox(
            "Fighter B", ids, index=min(1, len(ids) - 1), format_func=label
        )
    summaries = rankings(appearances).set_index("fighter_id")
    columns = [
        "fights",
        "wins",
        "losses",
        "other_results",
        "finish_win_rate",
        "height_inches",
        "reach_inches",
        "profile_weight_lbs",
        "mean_age",
        "mean_sig_per_minute",
        "mean_td_per_15",
        "mean_sub_att",
        "stats_fights",
    ]
    comparison = summaries.loc[[a, b], columns].T
    comparison.columns = [f"A: {names[a]}", f"B: {names[b]}"]
    st.dataframe(comparison, width="stretch")
    trend = appearances[appearances.fighter_id.isin([a, b])].copy()
    trend["Fighter"] = trend.fighter_id.map(label)
    chart(
        px.scatter(
            trend,
            x="event_date",
            y="sig_per_minute",
            color="Fighter",
            title="Fight-by-fight striking rate",
            labels={
                "event_date": "Fight date",
                "sig_per_minute": "Significant strikes / minute",
            },
        )
    )
    st.caption(
        "Mean age is age at selected fights. Height, reach, weight and stance are profile snapshots; weight is not historical weigh-in weight. Compare sample size and coverage before interpreting differences."
    )

elif page == "Fight Prediction / ML":
    metadata = json.loads(
        (ROOT / "outputs" / "tables" / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    metrics = load_table("model_metrics")
    st.info(
        "Historical analytical demonstration, not a betting system. Features use age and earlier-date UFC history. Current-fight statistics and profile career averages are excluded."
    )
    st.write(
        f"**Selected model:** {metadata['selected_model']} · selected by validation log loss, before test evaluation."
    )
    table(load_table("model_splits"))
    table(metrics)
    st.caption(
        "Train 1994–2019; validation 2020–2022; test 2023–snapshot end. Final models fit through 2022. Earlier test outcomes update later test histories, matching sequential replay; model parameters stay fixed."
    )
    roc = load_table("roc_curves")
    chart(
        px.line(
            roc,
            x="false_positive_rate",
            y="true_positive_rate",
            color="model",
            title="Held-out ROC curves",
            labels={
                "false_positive_rate": "False positive rate",
                "true_positive_rate": "True positive rate",
            },
        )
    )
    predictions = load_table("test_predictions")
    selected = predictions[predictions.model.eq(metadata["selected_model"])].copy()
    selected["label"] = (
        selected.event_date + " · " + selected.a_name + " vs " + selected.b_name
    )
    replay = st.selectbox(
        "Replay a held-out matchup",
        selected.fight_id.tolist(),
        format_func=lambda fid: selected.set_index("fight_id").loc[fid, "label"],
    )
    row = selected[selected.fight_id.eq(replay)].iloc[0]
    st.metric(f"Model probability: {row.a_name} wins", f"{row.p_a_wins:.1%}")
    if st.checkbox("Reveal recorded result"):
        st.write(
            "Recorded winner: **" + (row.a_name if row.a_won else row.b_name) + "**"
        )
    with st.expander("Feature effects and evaluation detail"):
        confusion = load_table("model_confusion").pivot(
            index="actual", columns="predicted", values="count"
        )
        chart(
            px.imshow(
                confusion,
                text_auto=True,
                x=["B wins", "A wins"],
                y=["B wins", "A wins"],
                title="Held-out confusion matrix",
                labels={"x": "Predicted", "y": "Recorded", "color": "Fights"},
                color_continuous_scale="Blues",
            )
        )
        st.write(
            "Logistic coefficients use standardized differences; correlated history features can make individual coefficients unstable."
        )
        table(load_table("model_coefficients"))
        table(load_table("model_subgroups"))
        table(load_table("model_calibration"))

else:
    findings = ROOT / "outputs" / "tables" / "insights.json"
    if findings.exists():
        for insight in json.loads(findings.read_text(encoding="utf-8")):
            st.subheader(insight["observation"])
            st.write(insight["evidence"])
            st.caption(insight["interpretation"])
    else:
        st.info("Run python -m src.reporting to refresh the research findings.")
    with st.expander("Scope and data limitations"):
        st.write(
            "The source mixes promotions, has no venue or reliable main-event field, lacks bonus recipient IDs, and includes present-day fighter profile metrics. Scope, missingness and denominators are documented in DATA_NOTES.md."
        )
        table(data["quality"].query("affected_rows > 0"))

st.divider()
st.caption(
    "UFC Fight Analytics · Source: Kaggle UFC Datasets 1994–2026 · Contact: dbui10@fordham.edu"
)
