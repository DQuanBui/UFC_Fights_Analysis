"""Cross-source reconciliation and refreshable audit evidence."""

import numpy as np
import pandas as pd

from .data_loader import TABLES, load_raw
from .feature_engineering import STATS


def audit_snapshot(data, raw=None):
    raw = raw if raw is not None else load_raw()
    fights = data["fights"]
    rounds = data["rounds"]
    appearances = data["appearances"]
    rows = []

    def add(check, count, interpretation):
        rows.append(
            dict(check=check, affected_rows=int(count), interpretation=interpretation)
        )

    for name, key, target, target_key in [
        ("fight", "event_id", "event", "event_id"),
        ("fight", "r_id", "fighter", "fighter_id"),
        ("fight", "b_id", "fighter", "fighter_id"),
        ("round", "fight_id", "fight", "fight_id"),
        ("fighter_bonus", "fight_id", "fight", "fight_id"),
    ]:
        add(
            f"Orphans {name}.{key}",
            (~raw[name][key].isin(raw[target][target_key])).sum(),
            "Expected zero",
        )
    for areas in [("head", "body", "leg"), ("distance", "clinch", "ground")]:
        for action, total in [("landed", "sig_landed"), ("atmp", "sig_atmp")]:
            columns = [f"sig_str_{action}_{x}" for x in areas]
            add(
                f"Strike decomposition {action} {areas}",
                rounds[columns].sum(axis=1, min_count=3).ne(rounds[total]).sum(),
                "Expected zero on observed statistics",
            )
    master = raw["master"].set_index("fight_id")
    for corner in ("r", "b"):
        observed = appearances.merge(
            fights[["fight_id", f"{corner}_id"]], on="fight_id", validate="many_to_one"
        )
        observed = observed[observed.fighter_id.eq(observed[f"{corner}_id"])].set_index(
            "fight_id"
        )
        for stat in STATS:
            name = f"{corner}_total_{stat}"
            if name in master:
                eligible = observed[stat].notna()
                mismatch = ~np.isclose(
                    observed.loc[eligible, stat],
                    master.loc[observed.index[eligible], name],
                )
                add(
                    "Master total reconciliation: " + name,
                    mismatch.sum(),
                    "Compare known complete reconstructed totals only",
                )
    for corner in ("r", "b"):
        mapping = {
            "fighter_name": "fighter_name",
            "fighter_nick_name": "fighter_nick_name",
            "height": "height",
            "weight_lbs": "weight_lbs",
            "reach_inches": "reach_inches",
            "stance": "stance",
            "dob": "dob",
            **{
                c: c
                for c in [
                    "slpm",
                    "str_acc",
                    "sapm",
                    "str_def",
                    "td_avg",
                    "td_acc",
                    "td_def",
                    "sub_avg",
                ]
            },
        }
        ids = master[f"{corner}_fighter_id"]
        dimension = raw["fighter"].set_index("fighter_id")
        for master_suffix, source_col in mapping.items():
            expected = ids.map(dimension[source_col])
            actual = master[f"{corner}_{master_suffix}"]
            equal = actual.eq(expected) | (actual.isna() & expected.isna())
            add(
                f"Master profile reconciliation: {corner}_{master_suffix}",
                (~equal).sum(),
                "Source profile join should match master",
            )
    add(
        "Unknown UFC round-stat fights",
        ((~fights.round_stats_complete) & fights.in_scope).sum(),
        "Retained with missing totals; biased toward early years",
    )
    add(
        "Noncontiguous partial round sequences",
        (fights.stats_round_count.notna() & ~fights.round_stats_complete).sum(),
        "Retain; exclude from complete-fight totals",
    )
    coverage = (
        fights.assign(
            missing_stats=~fights.round_stats_complete,
            logged_error=fights.fight_id.isin(raw["scrape_error"].entity_id),
        )
        .groupby(["fight_year", "is_ufc", "in_scope"])
        .agg(
            fights=("fight_id", "size"),
            missing_round_stats=("missing_stats", "sum"),
            scrape_errors=("logged_error", "sum"),
            complete_stats_share=("round_stats_complete", "mean"),
        )
        .reset_index()
    )
    pd.DataFrame(rows).to_csv(TABLES / "audit_checks.csv", index=False)
    coverage.to_csv(TABLES / "coverage_by_year.csv", index=False)
    fights.loc[
        ~fights.round_stats_complete,
        [
            "fight_id",
            "event_id",
            "event_name",
            "event_date",
            "is_ufc",
            "in_scope",
            "finish_round",
            "stats_round_count",
        ],
    ].to_csv(TABLES / "incomplete_round_stats.csv", index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from .pipeline import prepare

    checks = audit_snapshot(prepare(False))
    print(checks.query("affected_rows > 0").to_string(index=False))
