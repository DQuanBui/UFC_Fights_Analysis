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
