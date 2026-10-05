"""Exploratory effect sizes, paired comparisons and uncertainty estimates."""

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import cohort, save_tables


def wilson_interval(wins, n, confidence=0.95):
    if n == 0:
        return np.nan, np.nan
    z = stats.norm.ppf((1 + confidence) / 2)
    p = wins / n
    middle = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return middle - radius, middle + radius


def event_bootstrap(values, event_ids, repetitions=2000):
    """Resample entire events to preserve within-card dependence."""
    rows = pd.DataFrame({"value": values, "event_id": event_ids}).dropna()
    clusters = rows.groupby("event_id").value.agg(["sum", "count"]).to_numpy()
    if len(clusters) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(clusters), (repetitions, len(clusters)))
    sample = clusters[indices].sum(axis=1)
    return tuple(np.quantile(sample[:, 0] / sample[:, 1], [0.025, 0.975]))


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    )
    return adjusted


def export_statistics(data):
    fights, appearances = cohort(data)
    decisive = appearances[appearances.decisive]
    columns = [
        "age",
        "height_inches",
        "reach_inches",
        "sig_landed",
        "sig_accuracy",
        "td_success",
        "kd",
        "sub_att",
        "prior_ufc_fights",
    ]
    winners = decisive[decisive.won].set_index("fight_id")
    losers = decisive[decisive.lost].set_index("fight_id")
    rows = []
    for col in columns:
        pairs = pd.concat(
            [winners[col].rename("winner"), losers[col].rename("loser")], axis=1
        ).dropna()
        delta = pairs.winner - pairs.loser
        test = (
            stats.wilcoxon(delta, zero_method="wilcox") if delta.ne(0).any() else None
        )
        low, high = event_bootstrap(
            delta.to_numpy(), winners.loc[pairs.index, "event_id"].to_numpy()
        )
        rows.append(
            dict(
                metric=col,
                paired_fights=len(pairs),
                winner_mean=pairs.winner.mean(),
                loser_mean=pairs.loser.mean(),
                mean_paired_difference=delta.mean(),
                median_paired_difference=delta.median(),
                paired_standardized_effect=delta.mean() / delta.std(ddof=1)
                if delta.std(ddof=1)
                else np.nan,
                event_bootstrap_low=low,
                event_bootstrap_high=high,
                wilcoxon_p=test.pvalue if test else 1.0,
            )
        )
    paired = pd.DataFrame(rows)
    paired["holm_p"] = holm_adjust(paired.wilcoxon_p)
    advantage = []
    for col in ["age", "height_inches", "reach_inches"]:
        delta = fights[f"{col}_difference"]
        valid = fights.decisive & delta.notna() & delta.ne(0)
        subset = fights[valid]
        success = subset.winner_id.eq(
            subset.r_id.where(delta.loc[valid].gt(0), subset.b_id)
        ).astype(int)
        low, high = wilson_interval(success.sum(), len(success))
        clow, chigh = event_bootstrap(success.to_numpy(), subset.event_id.to_numpy())
        advantage.append(
            dict(
                attribute=col,
                n=len(success),
                advantaged_wins=success.sum(),
                win_rate=success.mean(),
                wilson_low=low,
                wilson_high=high,
                event_bootstrap_low=clow,
                event_bootstrap_high=chigh,
                excluded_equal_or_missing_or_unresolved=len(fights) - len(success),
            )
        )
    sufficient = fights.weight_class.value_counts().loc[lambda x: x.ge(100)].index
    table = pd.crosstab(
        fights.loc[fights.weight_class.isin(sufficient), "weight_class"],
        fights.loc[fights.weight_class.isin(sufficient), "outcome_group"],
    )
    table = table.reindex(columns=["KO/TKO", "Submission", "Decision"]).fillna(0)
    chi, p, dof, expected = stats.chi2_contingency(table)
    chi_table = pd.DataFrame(
        [
            dict(
                question="Division versus method among KO/submission/decision fights",
                n=table.to_numpy().sum(),
                chi_squared=chi,
                degrees_freedom=dof,
                p_value=p,
                min_expected=expected.min(),
                cramers_v=np.sqrt(
                    chi
                    / (
                        table.to_numpy().sum()
                        * min(table.shape[0] - 1, table.shape[1] - 1)
                    )
                ),
            )
        ]
    )
    unique = appearances.drop_duplicates("fighter_id")[
        ["height_inches", "reach_inches"]
    ].dropna()
    pearson = stats.pearsonr(unique.height_inches, unique.reach_inches)
    spearman = stats.spearmanr(unique.height_inches, unique.reach_inches)
    correlation = pd.DataFrame(
        [
            dict(
                question="Height versus reach, one row per observed UFC fighter",
                n=len(unique),
                pearson_r=pearson.statistic,
                pearson_p=pearson.pvalue,
                spearman_r=spearman.statistic,
                spearman_p=spearman.pvalue,
            )
        ]
    )
    five = fights[fights.standard_format & fights.scheduled_rounds.eq(5)]
    a = five.loc[five.title_fight.eq(1), "fight_duration_seconds"].dropna() / 60
    b = five.loc[five.title_fight.eq(0), "fight_duration_seconds"].dropna() / 60
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    title = pd.DataFrame(
        [
            dict(
                question="Title vs non-title among standard five-round fights",
                title_n=len(a),
                non_title_n=len(b),
                title_mean_minutes=a.mean(),
                non_title_mean_minutes=b.mean(),
                mean_difference_minutes=a.mean() - b.mean(),
                mann_whitney_p=u.pvalue,
                rank_biserial=2 * u.statistic / (len(a) * len(b)) - 1,
            )
        ]
    )
    tables = {
        "paired_statistics": paired,
        "advantage_uncertainty": pd.DataFrame(advantage),
        "division_association": chi_table,
        "physical_correlation": correlation,
        "title_statistics": title,
    }
    save_tables(tables)
    return tables
