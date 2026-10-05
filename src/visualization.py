"""Consistent publication figures and Plotly chart styling."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .analysis import (
    annual_summary,
    cohort,
    composition_adjustment,
    division_summary,
    performance_summary,
    round_hazard,
)
from .data_loader import ROOT, TABLES

NAVY = "#172B4D"
RED = "#C8383D"
TEAL = "#087F8C"
GOLD = "#BE872B"
COLORS = [RED, TEAL, NAVY, GOLD, "#8291A6", "#B47A9E"]
CHARTS = ROOT / "outputs" / "charts"


def style():
    sns.set_theme(style="whitegrid", palette=COLORS, font_scale=1.03)
    plt.rcParams.update(
        {
            "figure.facecolor": "#FAFAF7",
            "axes.facecolor": "#FAFAF7",
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.22,
            "savefig.dpi": 160,
            "figure.dpi": 110,
        }
    )


def finish(fig, title, subtitle=None):
    fig.suptitle(title, x=0.08, ha="left", fontsize=17, fontweight="bold", color=NAVY)
    if subtitle:
        fig.text(0.08, 0.91, subtitle, fontsize=10, color="#53627A")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.87])
    return fig


def history_chart(fights, appearances):
    style()
    annual = annual_summary(fights, appearances)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in zip(
        axes, ["events", "active_fighters"], ["Events", "Unique active fighters"]
    ):
        full = annual[~annual.partial_year]
        ax.plot(full.fight_year, full[col], color=RED, lw=2.5)
        partial = annual[annual.partial_year]
        ax.scatter(
            partial.fight_year,
            partial[col],
            facecolors="none",
            edgecolors=RED,
            s=65,
            label="Partial 2026",
        )
        ax.set(xlabel="Year", ylabel=label)
        ax.legend(frameon=False)
    return finish(
        fig,
        "UFC expanded in scale and participation",
        "1994–2026 snapshot • final year is incomplete",
    )


def outcome_chart(fights):
    style()
    rates = (
        pd.crosstab(fights.fight_year, fights.outcome_group, normalize="index") * 100
    )
    order = [
        c
        for c in ["KO/TKO", "Submission", "Decision", "Draw", "No contest", "Other"]
        if c in rates
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(
        rates.index, [rates[c] for c in order], labels=order, colors=COLORS, alpha=0.9
    )
    ax.set(xlabel="Year", ylabel="Share of all fights (%)", ylim=(0, 100))
    ax.legend(ncol=3, loc="upper left", frameon=True)
    return finish(
        fig,
        "How fights end has changed over time",
        "All results remain in the denominator; doctor stoppages count as KO/TKO",
    )


def division_chart(fights, appearances):
    style()
    table = (
        division_summary(fights, appearances)
        .query("fights >= 100")
        .sort_values("finish_rate")
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(table))
    ax.barh(y, table.ko_rate * 100, label="KO/TKO", color=RED)
    ax.barh(
        y,
        table.submission_rate * 100,
        left=table.ko_rate * 100,
        label="Submission",
        color=TEAL,
    )
    ax.set_yticks(
        y, [f"{r.weight_class}  (n={r.fights:,})" for r in table.itertuples()]
    )
    ax.set(xlabel="Finishes / all division fights (%)", xlim=(0, 100))
    ax.legend(frameon=False, loc="lower right")
    return finish(
        fig,
        "Divisions have distinct finishing profiles",
        "Minimum 100 fights • KO/TKO plus submissions define finishes",
    )


def hazard_chart(fights):
    style()
    table = round_hazard(fights)
    fig, ax = plt.subplots(figsize=(10, 5))
    for schedule, group in table.groupby("scheduled_rounds"):
        ax.plot(
            group.round_no,
            group.finish_risk * 100,
            marker="o",
            lw=2,
            label=f"{schedule}-round schedule",
        )
        for row in group.itertuples():
            ax.annotate(
                f"n={row.at_risk:,}",
                (row.round_no, row.finish_risk * 100),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    ax.set(
        xlabel="Round reached",
        ylabel="KO/submission in round / fights reaching round (%)",
        xticks=range(1, 6),
        ylim=(0, 37),
    )
    ax.legend(frameon=False)
    return finish(
        fig,
        "Conditional finish risk falls in later rounds",
        "Descriptive risk sets; standard five-minute schedules only",
    )


def performance_chart(appearances):
    style()
    table = performance_summary(appearances, "fight_year")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in zip(
        axes,
        ["sig_landed_per_minute", "td_per_15"],
        [
            "Significant strikes landed / fighter-minute",
            "Takedowns landed / 15 fighter-minutes",
        ],
    ):
        ax.plot(table.fight_year, table[col], lw=2.5)
        ax.set(xlabel="Year", ylabel=label)
    return finish(
        fig,
        "Striking and grappling intensity across the snapshot",
        "Pooled exposure rates • missing fight statistics are excluded",
    )


def physical_chart(appearances):
    style()
    unique = appearances.drop_duplicates("fighter_id").dropna(
        subset=["height_inches", "reach_inches"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hexbin(
        unique.height_inches, unique.reach_inches, gridsize=24, cmap="Blues", mincnt=1
    )
    axes[0].set(xlabel="Profile height (inches)", ylabel="Profile reach (inches)")
    pairs = appearances[appearances.decisive].copy()
    pairs["result"] = np.where(pairs.won, "Winner", "Loser")
    sns.violinplot(
        data=pairs,
        x="result",
        y="age",
        inner="quart",
        cut=0,
        color="#72AEB7",
        ax=axes[1],
    )
    axes[1].set(xlabel="", ylabel="Age at fight (years)")
    return finish(
        fig,
        "Physical traits describe different questions",
        "Left: unique fighters • right: fight appearances • associations are observational",
    )


def plotly_style(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Arial", color=NAVY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORS,
        margin=dict(l=20, r=20, t=55, b=40),
        legend_title_text="",
    )
    return fig


def export_charts(data):
    fights, appearances = cohort(data)
    CHARTS.mkdir(parents=True, exist_ok=True)
    figures = {
        "ufc_growth": history_chart(fights, appearances),
        "outcomes_over_time": outcome_chart(fights),
        "division_finishes": division_chart(fights, appearances),
        "round_finish_risk": hazard_chart(fights),
        "striking_grappling_trends": performance_chart(appearances),
        "physical_attributes": physical_chart(appearances),
    }
    style()
    geo = (
        fights.drop_duplicates("event_id").country.value_counts().head(12).sort_values()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    geo.plot.barh(ax=ax, color=TEAL)
    ax.set(xlabel="Events in snapshot", ylabel="Country or territory")
    figures["geography"] = finish(
        fig,
        "Where UFC events are represented",
        "Locations describe cities; venue names are unavailable",
    )
    table = composition_adjustment(fights).set_index("period")
    fig, ax = plt.subplots(figsize=(10, 5))
    (table[["all_division_decision_rate", "standardized_decision_rate"]] * 100).rename(
        columns={
            "all_division_decision_rate": "Observed mix",
            "standardized_decision_rate": "Fixed shared division mix",
        }
    ).plot.bar(ax=ax, color=[NAVY, TEAL], rot=0)
    ax.set(xlabel="", ylabel="Decision share (%)", ylim=(0, 65))
    ax.legend(frameon=False)
    figures["division_mix"] = finish(
        fig,
        "The division mix changes the interpretation",
        "Fixed pooled weights • at least 50 bouts per division in each comparison period",
    )
    paired = pd.read_csv(TABLES / "paired_statistics.csv").head(3)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.errorbar(
        paired.mean_paired_difference,
        range(3),
        xerr=[
            paired.mean_paired_difference - paired.event_bootstrap_low,
            paired.event_bootstrap_high - paired.mean_paired_difference,
        ],
        fmt="o",
        capsize=4,
        color=RED,
    )
    ax.set_yticks(range(3), ["Age (years)", "Height (inches)", "Reach (inches)"])
    ax.axvline(0, color=NAVY, lw=1)
    ax.set(xlabel="Mean winner minus loser difference")
    figures["paired_effects"] = finish(
        fig,
        "Small physical advantages, a larger age gap",
        "95% event-bootstrap intervals • units differ by row",
    )
    roc = pd.read_csv(TABLES / "roc_curves.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, rows in roc.groupby("model"):
        ax.plot(rows.false_positive_rate, rows.true_positive_rate, label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(frameon=False, fontsize=9)
    figures["model_roc"] = finish(
        fig,
        "Pre-fight models offer modest discrimination",
        "Untouched test period: 2023 through snapshot end",
    )
    calibration = pd.read_csv(TABLES / "model_calibration.csv")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot(
        calibration.mean_prediction, calibration.observed_a_win_rate, "o-", color=TEAL
    )
    ax.set(
        xlabel="Mean predicted A win probability",
        ylabel="Observed A win fraction",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    figures["model_calibration"] = finish(
        fig,
        "Probability calibration needs scrutiny",
        "Validation-selected model • eight quantile bins in the held-out test period",
    )
    confusion = pd.read_csv(TABLES / "model_confusion.csv").pivot(
        index="actual", columns="predicted", values="count"
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        xticklabels=["B wins", "A wins"],
        yticklabels=["B wins", "A wins"],
    )
    ax.set(xlabel="Predicted result at 0.5 threshold", ylabel="Recorded result")
    figures["model_confusion"] = finish(
        fig,
        "Where held-out predictions are right and wrong",
        "Validation-selected model • one prediction per decisive test fight",
    )
    for name, figure in figures.items():
        figure.savefig(CHARTS / f"{name}.png", bbox_inches="tight")
        plt.close(figure)
    return list(figures)
