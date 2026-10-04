"""Fight-level statistics and strictly earlier-date fighter history."""
import numpy as np
import pandas as pd

STATS = ['kd', 'sig_landed', 'sig_atmp', 'total_str_landed', 'total_str_atmp',
         'td_success', 'td_atmp', 'sub_att', 'rev', 'ctrl_seconds'] + [
             f'sig_str_{action}_{area}' for action in ('landed', 'atmp')
             for area in ('head', 'body', 'leg', 'distance', 'clinch', 'ground')]


def safe_ratio(numerator, denominator):
    return numerator / denominator.where(denominator.gt(0))


def fighter_appearances(fights, rounds):
    """Two rows per bout, with totals only where every expected round is observed."""
    base_cols = ['fight_id', 'event_id', 'event_name', 'event_date', 'fight_year', 'weight_class', 'gender',
                 'in_scope', 'is_ufc', 'era', 'title_fight', 'outcome_group', 'result_status', 'decisive',
                 'is_finish', 'is_ko', 'is_submission', 'is_decision', 'fight_duration_seconds', 'round_stats_complete']
    parts = []
    for corner, opponent in [('r', 'b'), ('b', 'r')]:
        part = fights[base_cols].copy()
        for attr in ('id', 'fighter_name', 'age', 'height_inches', 'reach_inches', 'weight_lbs', 'stance'):
            part['fighter_id' if attr == 'id' else attr] = fights[f'{corner}_{attr}']
        part['opponent_id'] = fights[f'{opponent}_id']
        part['won'] = fights.decisive & fights.winner_id.eq(fights[f'{corner}_id']).fillna(False)
        part['lost'] = fights.decisive & ~part.won
        for col in ('finish', 'ko', 'submission', 'decision'):
            part[f'{col}_win'] = part.won & part[f'is_{col}']
        parts.append(part)
    appearances = pd.concat(parts, ignore_index=True)
    grouped = rounds.groupby(['fight_id', 'fighter_id'])[STATS]
    totals = grouped.sum(min_count=1)
    counts = grouped.count()
    sizes = grouped.size()
    for col in STATS:
        totals[col] = totals[col].where(counts[col].eq(sizes))
    appearances = appearances.merge(totals, on=['fight_id', 'fighter_id'], how='left', validate='one_to_one')
    appearances.loc[~appearances.round_stats_complete, STATS] = np.nan
    appearances['sig_accuracy'] = safe_ratio(appearances.sig_landed, appearances.sig_atmp)
    appearances['td_accuracy'] = safe_ratio(appearances.td_success, appearances.td_atmp)
    appearances['sig_per_minute'] = safe_ratio(appearances.sig_landed, appearances.fight_duration_seconds/60)
    appearances['td_per_15'] = 15*safe_ratio(appearances.td_success, appearances.fight_duration_seconds/60)
    appearances['control_share'] = safe_ratio(appearances.ctrl_seconds, appearances.fight_duration_seconds)
    defense = appearances[['fight_id', 'fighter_id', 'td_success', 'td_atmp']].rename(
        columns={'fighter_id':'opponent_id', 'td_success':'opponent_td_success', 'td_atmp':'opponent_td_atmp'})
    appearances = appearances.merge(defense, on=['fight_id', 'opponent_id'], validate='one_to_one')
    appearances['td_defense'] = 1-safe_ratio(appearances.opponent_td_success, appearances.opponent_td_atmp)
    return appearances


def add_history(appearances):
    """Exclude every bout on the current date, including tournament bouts."""
    source = appearances.loc[appearances.is_ufc].copy()
    source['ufc_fights'] = 1
    source['wins'] = source.won.astype(int)
    source['losses'] = source.lost.astype(int)
    source['ko_wins'] = source.ko_win.astype(int)
    source['submission_wins'] = source.submission_win.astype(int)
    source['title_fights'] = source.title_fight
    valid_stats = source.round_stats_complete & source.fight_duration_seconds.gt(0)
    source['observed_minutes'] = source.fight_duration_seconds.div(60).where(valid_stats, 0)
    for col in ('sig_landed', 'sig_atmp', 'td_success', 'td_atmp'):
        source['observed_'+col] = source[col].where(valid_stats, 0).fillna(0)
    counts = ['ufc_fights', 'wins', 'losses', 'ko_wins', 'submission_wins', 'title_fights',
              'observed_minutes', 'observed_sig_landed', 'observed_sig_atmp', 'observed_td_success', 'observed_td_atmp']
    daily = source.groupby(['fighter_id', 'event_date'], as_index=False)[counts].sum().sort_values(['fighter_id', 'event_date'])
    for col in counts:
        daily['prior_'+col] = daily.groupby('fighter_id')[col].cumsum() - daily[col]
    streaks = []
    for _, group in daily.groupby('fighter_id', sort=False):
        streak = 0
        for row in group.itertuples():
            streaks.append((row.Index, streak))
            # Mixed results or unresolved results on a tournament date reset the streak.
            streak = streak+row.wins if row.wins == row.ufc_fights else 0
    daily['prior_win_streak'] = pd.Series(dict(streaks))
    daily['prior_win_rate'] = safe_ratio(daily.prior_wins, daily.prior_wins+daily.prior_losses)
    daily['prior_ko_rate'] = safe_ratio(daily.prior_ko_wins, daily.prior_ufc_fights)
    daily['prior_submission_rate'] = safe_ratio(daily.prior_submission_wins, daily.prior_ufc_fights)
    daily['prior_sig_per_minute'] = safe_ratio(daily.prior_observed_sig_landed, daily.prior_observed_minutes)
    daily['prior_sig_accuracy'] = safe_ratio(daily.prior_observed_sig_landed, daily.prior_observed_sig_atmp)
    daily['prior_td_per_15'] = 15*safe_ratio(daily.prior_observed_td_success, daily.prior_observed_minutes)
    daily['prior_td_accuracy'] = safe_ratio(daily.prior_observed_td_success, daily.prior_observed_td_atmp)
    prior_cols = [c for c in daily if c.startswith('prior_')]
    return appearances.merge(daily[['fighter_id', 'event_date']+prior_cols], on=['fighter_id', 'event_date'], how='left', validate='many_to_one')


def build_features(clean):
    result = dict(clean)
    result['appearances'] = add_history(fighter_appearances(clean['fights'], clean['rounds']))
    fights = clean['fights'].copy()
    prior = [c for c in result['appearances'] if c.startswith('prior_')]
    for corner in ('r', 'b'):
        subset = result['appearances'][['fight_id', 'fighter_id']+prior].rename(
            columns={'fighter_id':f'{corner}_id', **{c:f'{corner}_{c}' for c in prior}})
        fights = fights.merge(subset, on=['fight_id', f'{corner}_id'], validate='one_to_one')
    result['fights'] = fights
    return result
