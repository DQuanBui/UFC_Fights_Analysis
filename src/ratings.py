"""UFC-only Elo and activity history, frozen before each event date."""

from collections import defaultdict

import numpy as np
import pandas as pd


def expected_score(rating, opponent_rating):
    return 1 / (1 + 10 ** ((opponent_rating - rating) / 400))


def pre_fight_state(fights, k=32.0, initial=1500.0):
    """Replay all UFC dates, including 1993, without same-day information leakage.

    Wins/losses and draws update zero-sum ratings. No contests do not update
    ratings, but all appearances update last-fight dates and opponent history.
    Same-day tournament deltas are summed only after all pre-date rows exist.
    Ratings are global across divisions, with no inactivity decay or reset.
    """
    if not np.isfinite(k) or k <= 0 or not np.isfinite(initial):
        raise ValueError("k must be positive and initial rating must be finite")
    source = fights.loc[fights.is_ufc].sort_values(["event_date", "fight_id"])
    if source.fight_id.duplicated().any() or source.event_date.isna().any():
        raise ValueError("UFC fights require unique IDs and known dates")
    if source.r_id.eq(source.b_id).any():
        raise ValueError("A fighter cannot face themself")
    ratings = defaultdict(lambda: float(initial))
    counts = defaultdict(int)
    opponent_totals = defaultdict(float)
    previous_dates = {}
    rows = []
    for date, daily in source.groupby("event_date", sort=True):
        changes = defaultdict(float)
        encounters = []
        for bout in daily.itertuples():
            red, blue = bout.r_id, bout.b_id
            probability = expected_score(ratings[red], ratings[blue])
            row = dict(fight_id=bout.fight_id, event_date=date, p_red_elo=probability)
            for corner, fighter in [("r", red), ("b", blue)]:
                row[f"{corner}_elo"] = ratings[fighter]
                row[f"{corner}_days_since"] = (
                    (date - previous_dates[fighter]).days
                    if fighter in previous_dates
                    else np.nan
                )
                row[f"{corner}_prior_bouts"] = counts[fighter]
                row[f"{corner}_prior_opponent_elo"] = (
                    opponent_totals[fighter] / counts[fighter]
                    if counts[fighter]
                    else np.nan
                )
            rows.append(row)
            if bout.decisive:
                if bout.winner_id not in (red, blue):
                    raise ValueError("Decisive result has no valid winner")
                score = float(bout.winner_id == red)
            elif bout.result_status == "draw":
                score = 0.5
            else:
                score = None
            if score is not None:
                change = k * (score - probability)
                changes[red] += change
                changes[blue] -= change
            encounters.extend([(red, ratings[blue]), (blue, ratings[red])])
        for fighter, change in changes.items():
            ratings[fighter] += change
        for fighter, opponent_elo in encounters:
            previous_dates[fighter] = date
            counts[fighter] += 1
            opponent_totals[fighter] += opponent_elo
    return pd.DataFrame(rows)
