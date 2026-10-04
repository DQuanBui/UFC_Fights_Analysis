"""Research tables with explicit units, denominators, and sample sizes."""
import numpy as np
import pandas as pd
from .data_loader import TABLES
from .feature_engineering import safe_ratio


def cohort(data):
    fights = data['fights'].loc[lambda x:x.in_scope].copy()
    appearances = data['appearances'].loc[lambda x:x.in_scope].copy()
    return fights, appearances


def annual_summary(fights, appearances):
    result = fights.groupby('fight_year').agg(fights=('fight_id','size'), events=('event_id','nunique'),
        finish_rate=('is_finish','mean'), ko_rate=('is_ko','mean'), submission_rate=('is_submission','mean'),
        decision_rate=('is_decision','mean'), title_rate=('title_fight','mean'),
        duration_minutes=('fight_duration_seconds',lambda x:x.mean()/60), stats_coverage=('round_stats_complete','mean'))
    result['active_fighters'] = appearances.groupby('fight_year').fighter_id.nunique()
    result['women_bout_share'] = fights.assign(women=fights.gender.eq('Women')).groupby('fight_year').women.mean()
    result['divisions'] = fights.groupby('fight_year').weight_class.nunique()
    event_rows = fights.drop_duplicates('event_id')
    result['known_location_events'] = event_rows.groupby('fight_year').country.count()
    result['countries_or_territories'] = event_rows.groupby('fight_year').country.nunique()
    result['international_event_share'] = event_rows[event_rows.country.notna()].assign(
        international=lambda x:x.country.ne('USA')).groupby('fight_year').international.mean()
    result['events_growth_pct'] = result.events.pct_change()*100
    result['partial_year'] = result.index == fights.fight_year.max()
    result.loc[result.partial_year, 'events_growth_pct'] = np.nan
    result['fights_per_event'] = result.fights / result.events
    return result.reset_index()


def event_summary(fights):
    keys=['event_id','event_name','event_date','event_year','location','country','city','event_category']
    return fights.groupby(keys, dropna=False).agg(fights=('fight_id','size'), finishes=('is_finish','sum'),
        finish_rate=('is_finish','mean'), title_fights=('title_fight','sum'),
        duration_minutes=('fight_duration_seconds',lambda x:x.mean()/60)).reset_index()


def export_history(data):
    fights, appearances = cohort(data)
    tables = {'annual':annual_summary(fights, appearances), 'events':event_summary(fights)}
    tables['geography'] = tables['events'].groupby('country',dropna=False).agg(
        events=('event_id','nunique'), first_year=('event_year','min'), last_year=('event_year','max')).reset_index().sort_values('events',ascending=False)
    tables['cities'] = tables['events'].groupby(['country','city'],dropna=False).size().rename('events').reset_index().sort_values('events',ascending=False)
    tables['event_categories'] = tables['events'].groupby('event_category').agg(
        events=('event_id','size'), mean_fights=('fights','mean'), mean_finish_rate=('finish_rate','mean')).reset_index()
    save_tables(tables)
    return tables


def save_tables(tables):
    TABLES.mkdir(parents=True,exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(TABLES / f'{name}.csv',index=False)


def fighter_summary(appearances, minimum_fights=10):
    """Percentages use decisive fights for wins, all appearances for finish wins."""
    group = appearances.groupby(['fighter_id','fighter_name'])
    result = group.agg(fights=('fight_id','size'), wins=('won','sum'), losses=('lost','sum'),
        ko_wins=('ko_win','sum'), submission_wins=('submission_win','sum'),
        decision_wins=('decision_win','sum'), finish_wins=('finish_win','sum'),
        title_fights=('title_fight','sum'), first_fight=('event_date','min'), last_fight=('event_date','max'),
        active_years=('fight_year','nunique'), mean_age=('age','mean'), height_inches=('height_inches','first'),
        reach_inches=('reach_inches','first'), profile_weight_lbs=('weight_lbs','first'),
        snapshot_stance=('stance','first'), stats_fights=('sig_landed','count'))
    result['decisive_fights'] = result.wins + result.losses
    result['other_results'] = result.fights - result.decisive_fights
    result['win_rate'] = safe_ratio(result.wins,result.decisive_fights)
    result['finish_win_rate'] = safe_ratio(result.finish_wins,result.fights)
    result['finish_share_of_wins'] = safe_ratio(result.finish_wins,result.wins)
    result['eligible_percentage_rank'] = result.decisive_fights.ge(minimum_fights)
    result['observed_career_years'] = (result.last_fight-result.first_fight).dt.days/365.2425
    streaks = {}
    for key, rows in group:
        running = best = 0
        for _, daily in rows.groupby('event_date',sort=True):
            running = running+len(daily) if daily.won.all() else 0
            best = max(best,running)
        streaks[key] = best
    result['longest_win_streak_conservative'] = pd.Series(streaks)
    for col in ('sig_per_minute','td_per_15','sig_accuracy','td_accuracy','sub_att'):
        result['mean_'+col] = group[col].mean()
    return result.reset_index().sort_values(['wins','fights'],ascending=False)


def division_summary(fights, appearances):
    result = fights.groupby('weight_class').agg(fights=('fight_id','size'),
        finish_rate=('is_finish','mean'), ko_rate=('is_ko','mean'), submission_rate=('is_submission','mean'),
        decision_rate=('is_decision','mean'), duration_minutes=('fight_duration_seconds',lambda x:x.mean()/60),
        stats_coverage=('round_stats_complete','mean'))
    group = appearances.groupby('weight_class')
    result['unique_fighters'] = group.fighter_id.nunique()
    for col in ('age','height_inches','reach_inches','sig_per_minute','td_per_15','sig_accuracy','td_accuracy','kd','sub_att','control_share'):
        result['mean_'+col] = group[col].mean()
        result[col+'_n'] = group[col].count()
    return result.reset_index().sort_values('fights',ascending=False)


def attribute_advantage(fights, attribute, group='weight_class'):
    delta = fights[f'{attribute}_difference']
    valid = fights.decisive & delta.notna() & delta.ne(0)
    rows = fights.loc[valid,[group,'winner_id','r_id','b_id']].copy()
    advantage_id = fights.r_id.where(delta.gt(0),fights.b_id)
    rows['advantaged_won'] = rows.winner_id.eq(advantage_id.loc[valid])
    result = rows.groupby(group).advantaged_won.agg(['size','mean']).reset_index()
    result.columns=[group,'unequal_pairs','advantaged_win_rate']
    result['attribute']=attribute
    return result


def export_fighters(data):
    fights, appearances = cohort(data)
    tables={'fighters':fighter_summary(appearances), 'divisions':division_summary(fights,appearances)}
    tables['physical_advantage']=pd.concat([attribute_advantage(fights,c) for c in ['age','height_inches','reach_inches']],ignore_index=True)
    tables['stance']=appearances.groupby('stance',dropna=False).agg(appearances=('fight_id','size'),
        fighters=('fighter_id','nunique'), wins=('won','sum'), decisive=('decisive','sum'),
        finish_wins=('finish_win','sum')).reset_index()
    tables['stance']['win_rate']=safe_ratio(tables['stance'].wins,tables['stance'].decisive)
    tables['stance_by_year']=appearances.groupby(['fight_year','stance'],dropna=False).size().rename('appearances').reset_index()
    tables['stance_by_division']=appearances.groupby(['weight_class','stance'],dropna=False).size().rename('appearances').reset_index()
    tables['age_profile']=appearances.assign(age_band=pd.cut(appearances.age,[16,23,28,33,38,65],right=False)).groupby('age_band',observed=True).agg(
        appearances=('fight_id','size'), wins=('won','sum'), decisive=('decisive','sum')).reset_index()
    tables['age_profile']['win_rate']=safe_ratio(tables['age_profile'].wins,tables['age_profile'].decisive)
    save_tables(tables)
    return tables
