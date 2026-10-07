"""Evidence figures for the question-led notebook extensions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from .data_loader import TABLES
from .visualization import CHARTS, GOLD, NAVY, RED, TEAL, finish, style


def read(name):
    return pd.read_csv(TABLES / f"deep_{name}.csv")


def research_chart(number):
    style()
    fig, ax = plt.subplots(figsize=(11, 6))
    if number == "01":
        rows = read("missingness").set_index("period")
        cols = ["age_missing", "reach_missing", "control_missing", "statistics_missing"]
        sns.heatmap(
            rows[cols] * 100,
            annot=True,
            fmt=".1f",
            cmap="Reds",
            vmin=0,
            vmax=100,
            xticklabels=["Age", "Reach", "Control", "Round statistics"],
            cbar_kws={"label": "Missing appearances (%)"},
            ax=ax,
        )
        ax.set(xlabel="Measure", ylabel="Era")
        ax.tick_params(axis="y", rotation=0)
        title, subtitle = (
            "Missingness is concentrated in early UFC history",
            "Percentage of fighter appearances; repeated athletes are counted at each bout",
        )
    elif number == "02":
        rows = (
            read("rate_weighting")
            .query("fighter_fights >= 200")
            .sort_values("difference")
        )
        ax.hlines(
            rows.weight_class,
            rows.exposure_weighted_rate,
            rows.equal_fight_weighted_rate,
            color="#BAC2CF",
            lw=3,
        )
        ax.scatter(
            rows.exposure_weighted_rate,
            rows.weight_class,
            label="Total strikes / total minutes",
            color=TEAL,
        )
        ax.scatter(
            rows.equal_fight_weighted_rate,
            rows.weight_class,
            label="Mean of appearance rates",
            color=RED,
        )
        ax.set(xlabel="Significant strikes landed per fighter-minute", ylabel="")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9, ncol=2)
        title, subtitle = (
            "The denominator changes the striking-rate comparison",
            "Divisions with at least 200 fighter appearances; observed statistics only",
        )
    elif number == "03":
        rows = read("growth_decomposition")
        x = np.arange(len(rows))
        ax.bar(
            x - 0.18,
            rows.event_count_contribution,
            0.36,
            label="More events",
            color=RED,
        )
        ax.bar(
            x + 0.18,
            rows.card_size_contribution,
            0.36,
            label="More fights per event",
            color=TEAL,
        )
        ax.set_xticks(x, rows.start.astype(str) + " to " + rows.end.astype(str))
        ax.set(
            ylabel="Contribution to change in annual fight count", xlabel="Comparison"
        )
        ax.legend()
        title, subtitle = (
            "Event expansion drove most of the 2010-2015 growth",
            "Symmetric decomposition: the two contributions sum exactly to total fight growth",
        )
    elif number == "04":
        rows = read("career_continuation")
        labels = [
            f"Debut {'win' if str(v).lower() == 'true' else 'loss'} (n={n:,})"
            for v, n in zip(rows.debut_win, rows.n)
        ]
        ax.errorbar(
            rows.rate,
            labels,
            xerr=[rows.rate - rows.low, rows.high - rows.rate],
            fmt="o",
            color=TEAL,
            capsize=5,
            ms=9,
        )
        ax.set(xlim=(0, 1), xlabel="Share reaching a third UFC bout within 730 days")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        title, subtitle = (
            "Early career continuation differs sharply by debut result",
            "Equal 730-day follow-up; decisive debuts since 2000; Wilson 95% intervals; association only",
        )
    elif number == "05":
        rows = read("decision_decomposition").sort_values("total_contribution")
        y = np.arange(len(rows))
        ax.barh(
            y - 0.18,
            rows.within_contribution * 100,
            0.36,
            label="Within-division change",
            color=TEAL,
        )
        ax.barh(
            y + 0.18,
            rows.mix_contribution * 100,
            0.36,
            label="Division-mix change",
            color=RED,
        )
        ax.set_yticks(y, rows.weight_class)
        ax.axvline(0, color=NAVY, lw=1)
        ax.set(xlabel="Contribution to decision-rate change (percentage points)")
        ax.legend(fontsize=9)
        title, subtitle = (
            "Division mix and within-division changes pull in different directions",
            "2010-2019 to 2020-2025; shared divisions with at least 50 bouts in each period",
        )
    elif number == "06":
        rows = read("opening_round_leads")
        labels = [
            f"{label} (n={n:,})"
            for label, n in zip(
                ["Significant strikes", "Takedowns", "Control time"],
                rows.eligible_fights,
            )
        ]
        ax.errorbar(
            rows.leader_win_rate,
            labels,
            xerr=[rows.leader_win_rate - rows.low, rows.high - rows.leader_win_rate],
            fmt="o",
            color=TEAL,
            capsize=5,
            ms=9,
        )
        ax.axvline(0.5, color=GOLD, ls="--", label="50% reference")
        ax.set(
            xlim=(0.45, 0.75),
            xlabel="Opening-round statistical leader's eventual win rate",
        )
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        title, subtitle = (
            "First-round leads are associated with the eventual result",
            "Only decisive bouts continuing into round 2; ties/missing excluded; Wilson 95% intervals",
        )
    elif number == "07":
        rows = read("adjusted_attributes")
        ax.errorbar(
            rows.odds_ratio,
            [
                "Age (+5 years)",
                "Height (+2 inches)",
                "Reach (+2 inches)",
                "Prior UFC experience (+5 fights)",
            ],
            xerr=[
                rows.odds_ratio - rows.event_bootstrap_low,
                rows.event_bootstrap_high - rows.odds_ratio,
            ],
            fmt="o",
            color=TEAL,
            capsize=5,
            ms=9,
        )
        ax.axvline(1, color=GOLD, ls="--")
        ax.set(xlabel="Adjusted win odds ratio (1 = no association)", xlim=(0.6, 1.2))
        title, subtitle = (
            "Reach retains a modest association after adjustment",
            "7,195 complete-case bouts, 2000-2025; division/decade adjusted; 250 event-bootstrap replicates",
        )
    elif number == "08":
        rows = read("feature_ablation_folds")
        for name, group in rows.groupby("features"):
            ax.plot(group.year, group.log_loss, marker="o", label=name)
        ax.set(
            xlabel="Development evaluation year",
            ylabel="Log loss (lower is better)",
            xticks=sorted(rows.year.unique()),
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=9, ncol=2)
        title, subtitle = (
            "Historical features help, but gains vary by year",
            "Fit on earlier years only; same logistic specification; no 2023-2026 holdout tuning",
        )
    elif number == "09":
        rows = read("bonus_categories")
        for offset, category, color in [
            (-0.18, "Fight of the Night", TEAL),
            (0.18, "Performance of the Night", RED),
        ]:
            part = (
                rows[rows.category.eq(category)]
                .set_index("outcome")
                .loc[["Decision", "Finish"]]
            )
            ax.bar(
                np.arange(2) + offset,
                part.rate,
                0.36,
                label=category,
                color=color,
                yerr=[part.rate - part.low, part.high - part.rate],
                capsize=4,
            )
        ax.set(
            xticks=[0, 1],
            xticklabels=["Decisions", "Finishes"],
            ylabel="Share of fights with recorded category",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.legend()
        title, subtitle = (
            "Bonus categories capture different fight profiles",
            "2015-2025 UFC fights; Wilson 95% intervals; categories may overlap; recipient IDs unavailable",
        )
    elif number == "10":
        rows = read("elo_folds")
        for name, color in [
            ("Raw Elo K=32", GOLD),
            ("Historical logistic", TEAL),
            ("History plus Elo", RED),
        ]:
            part = rows[rows.model.eq(name)]
            ax.plot(part.year, part.log_loss, marker="o", color=color, label=name)
        ax.set(
            xlabel="Development evaluation year",
            ylabel="Log loss (lower is better)",
            xticks=sorted(rows.year.unique()),
        )
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
        title, subtitle = (
            "Elo adds a small, inconsistent increment to historical features",
            "Expanding-year evaluation; fixed K=32; 2023-2026 holdout excluded from this experiment",
        )
    elif number == "11":
        rows = read("layoff_matched")
        labels = [f"{label}\n(n={int(n):,})" for label, n in zip(rows.cohort, rows.n)]
        ax.errorbar(
            rows.win_rate,
            labels,
            xerr=[rows.win_rate - rows.event_low, rows.event_high - rows.win_rate],
            fmt="o",
            color=TEAL,
            capsize=5,
            ms=9,
        )
        ax.axvline(0.5, color=GOLD, ls="--", label="50% reference")
        ax.set(xlim=(0.35, 0.65), xlabel="Longer-rest fighter's win rate")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.margins(y=0.3)
        ax.legend(loc="lower right")
        title, subtitle = (
            "The rest-gap association weakens with age and rating restrictions",
            "2000-2025 decisive bouts; both fighters returning; event-bootstrap 95% intervals; observational",
        )
    elif number == "12":
        rows = read("generalization_familiarity")
        rows = rows[rows.reportable].sort_values("accuracy")
        labels = [f"{label}\n(n={int(n):,})" for label, n in zip(rows.group, rows.n)]
        ax.errorbar(
            rows.accuracy,
            labels,
            xerr=[rows.accuracy - rows.event_low, rows.event_high - rows.accuracy],
            fmt="o",
            color=TEAL,
            capsize=5,
            ms=9,
        )
        ax.scatter(
            rows.mean_confidence,
            labels,
            color=RED,
            marker="x",
            s=70,
            label="Mean predicted-winner confidence",
        )
        ax.set(xlim=(0.45, 0.75), xlabel="Observed accuracy / mean confidence")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.margins(y=0.3)
        ax.legend(loc="lower right", fontsize=9)
        title, subtitle = (
            "Fighter familiarity changes the evaluation population",
            "Frozen pre-2023 fitting membership; 2023-2026 held-out predictions; event-bootstrap 95% intervals",
        )
    else:
        plt.close(fig)
        raise ValueError(f"Unknown research chapter: {number}")
    return finish(fig, title, subtitle)


def export_research_charts():
    from .research_notebooks import SECTIONS

    CHARTS.mkdir(parents=True, exist_ok=True)
    for number in sorted(SECTIONS):
        fig = research_chart(number)
        fig.savefig(CHARTS / f"research_{number}.png")
        plt.close(fig)


if __name__ == "__main__":
    export_research_charts()
