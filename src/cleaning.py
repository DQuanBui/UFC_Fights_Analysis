"""Auditable source normalization with explicit scope and missingness."""
import re
import numpy as np
import pandas as pd
from .data_loader import KEYS

DIVISIONS = ["Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
             "Women's Featherweight", "Women's Atomweight", 'Super Heavyweight',
             'Light Heavyweight', 'Heavyweight', 'Middleweight', 'Welterweight',
             'Lightweight', 'Featherweight', 'Bantamweight', 'Flyweight', 'Open Weight', 'Catch Weight']


def parse_clock(value):
    """Parse elapsed M:SS, preserving unknown values rather than making zeros."""
    match = re.fullmatch(r'(\d+):([0-5]\d)', str(value).strip())
    return int(match[1])*60 + int(match[2]) if match else np.nan


def round_schedule(value, rounds=1):
    match = re.search(r'\(([\d-]+)\)', str(value))
    if not match:
        return None
    schedule = [int(x)*60 for x in match[1].split('-')]
    if str(value).startswith('Unlimited'):
        schedule = schedule * max(1, int(rounds))
    return schedule


def fight_duration(time_format, ending_round, time):
    elapsed = parse_clock(time)
    if pd.isna(ending_round) or pd.isna(elapsed) or ending_round < 1:
        return np.nan
    if time_format == 'No Time Limit':
        return elapsed if ending_round == 1 else np.nan
    schedule = round_schedule(time_format, ending_round)
    r = int(ending_round)
    if schedule is None or r > len(schedule) or elapsed > schedule[r-1]:
        return np.nan
    return sum(schedule[:r-1]) + elapsed


def parse_height(value):
    match = re.fullmatch(r'''(\d+)'\s*(\d+)"''', str(value).strip())
    return int(match[1])*12 + int(match[2]) if match and int(match[2]) < 12 else np.nan


def is_ufc_event(name):
    """Include UFC cards and televised TUF finales; exclude feeder competitions."""
    return bool(re.match(r'^(UFC\b|Noche UFC\b|The Ultimate Fighter\b|Ultimate Ultimate\b|Ultimate Japan\b)', name)
                or name == 'Ortiz vs Shamrock 3: The Final Chapter')


def normalize_division(value):
    for division in DIVISIONS:
        if division.lower() in value.lower():
            if "Women" in value and "Women" not in division:
                continue
            return division
    return {'Superfight': 'Open Weight'}.get(value, 'Other / unspecified')


def clean_tables(raw):
    """Return all fight rows plus clean dimensions, long rounds, and a quality ledger."""
    issues = []

    def report(check, n, action, severity='info'):
        issues.append(dict(check=check, affected_rows=int(n), severity=severity, action=action))

    for name, frame in raw.items():
        if frame.duplicated(KEYS[name]).any() or frame[KEYS[name]].isna().any().any():
            raise ValueError(f'{name}: nonunique or missing primary key')
    events = raw['event'].copy()
    for col in ('event_name', 'location'):
        events[col] = events[col].astype('string').str.strip().str.replace(r'\s+', ' ', regex=True)
    events['event_date'] = pd.to_datetime(events.pop('date'), format='%Y-%m-%d', errors='coerce')
    report('Malformed event dates', events.event_date.isna().sum(), 'Retain; exclude from dated analyses', 'warning')
    events['is_ufc'] = events.event_name.map(is_ufc_event)
    events['country'] = events.location.str.rsplit(',', n=1).str[-1].str.strip()
    events['city'] = events.location.str.split(',').str[0].str.strip()
    events['event_category'] = np.select([
        events.event_name.str.match(r'^UFC \d+').fillna(False),
        events.event_name.str.contains('Ultimate Fighter', case=False).fillna(False),
        events.is_ufc], ['Numbered UFC', 'TUF finale', 'Other UFC card'], default='Other promotion / feeder')
    events['event_year'] = events.event_date.dt.year
    report('Non-UFC event records', (~events.is_ufc).sum(), 'Retain in audit; exclude from headline UFC scope')
    report('Missing event locations', events.location.isna().sum(), 'Unknown; exclude from geographic denominators')
    report('Duplicate normalized event names', events.event_name.duplicated(False).sum(), 'Keep event IDs; do not merge names')

    fighters = raw['fighter'].copy()
    fighters['fighter_name'] = fighters.fighter_name.str.strip().str.replace(r'\s+', ' ', regex=True)
    fighters['height_inches'] = fighters.height.map(parse_height)
    fighters['dob'] = pd.to_datetime(fighters.dob, format='%Y-%m-%d', errors='coerce')
    report('Malformed nonmissing height', (fighters.height.notna() & fighters.height_inches.isna()).sum(), 'Set parsed measure to missing')
    report('Malformed nonmissing DOB', (raw['fighter'].dob.notna() & fighters.dob.isna()).sum(), 'Set parsed date to missing')
    for col, low, high in [('height_inches', 48, 96), ('reach_inches', 45, 100), ('weight_lbs', 80, 800)]:
        values = pd.to_numeric(fighters[col], errors='coerce')
        invalid = values.notna() & ~values.between(low, high)
        report(f'Implausible {col}', invalid.sum(), 'Retain source; analytical measure set to missing', 'warning')
        fighters[col] = values.mask(invalid)
    report('Historical profile weight above 400 lb', fighters.weight_lbs.gt(400).sum(),
           'Preserve flagged historical open-weight outliers; profile weight is not a weigh-in')
    report('Repeated fighter names across IDs', fighters.fighter_name.duplicated(False).sum(), 'Keep distinct IDs; use ID suffix in selectors')
    for col in ('str_acc', 'str_def', 'td_acc', 'td_def'):
        report(f'Invalid snapshot percentage {col}', (~fighters[col].between(0, 100)).sum(), 'Snapshot metrics excluded from modeling')

    fights = raw['fight'].copy().rename(columns={'round': 'finish_round', 'time': 'finish_time', 'weight_class': 'weight_class_raw'})
    if (~fights.event_id.isin(events.event_id)).any():
        raise ValueError('Orphan fight event IDs')
    if (~fights.r_id.isin(fighters.fighter_id) | ~fights.b_id.isin(fighters.fighter_id)).any():
        raise ValueError('Orphan fighter IDs')
    fights = fights.merge(events, on='event_id', how='left', validate='many_to_one')
    fights['weight_class'] = fights.weight_class_raw.map(normalize_division)
    fights['gender'] = np.where(fights.weight_class_raw.str.contains('Women'), 'Women', 'Men / unspecified')
    fights['fight_year'] = fights.event_date.dt.year
    fights['fight_month'] = fights.event_date.dt.month
    fights['in_scope'] = fights.is_ufc & fights.fight_year.between(1994, 2026)
    fights['fight_duration_seconds'] = [fight_duration(a, b, c) for a, b, c in
                                      zip(fights.time_format, fights.finish_round, fights.finish_time)]
    fights['scheduled_rounds'] = fights.time_format.map(lambda x: len(round_schedule(x) or []) or np.nan)
    fights['standard_format'] = fights.time_format.isin(['3 Rnd (5-5-5)', '5 Rnd (5-5-5-5-5)'])
    valid_winner = fights.winner_id.eq(fights.r_id) | fights.winner_id.eq(fights.b_id)
    fights['decisive'] = fights.result_status.eq('win') & valid_winner.fillna(False)
    report('Invalid or absent winner on win record', (fights.result_status.eq('win') & ~valid_winner.fillna(False)).sum(), 'Exclude from winner models', 'warning')
    report('Same fighter in both corners', fights.r_id.eq(fights.b_id).sum(), 'Requires source review', 'warning')
    report('Invalid duration or schedule', fights.fight_duration_seconds.isna().sum(), 'Keep fight; omit duration and per-minute metrics', 'warning')
    pair = fights.apply(lambda r: '|'.join(sorted([r.r_id, r.b_id])), axis=1)
    report('Repeated event and fighter pairing', pd.DataFrame({'event': fights.event_id, 'pair': pair}).duplicated(keep=False).sum(), 'Keep IDs; possible tournament rematches, never drop automatically')
    fights['outcome_group'] = np.select([
        fights.result_status.eq('no_contest'), fights.result_status.eq('draw'),
        fights.method.isin(['KO/TKO', "TKO - Doctor's Stoppage"]),
        fights.method.eq('Submission'), fights.method.str.startswith('Decision')],
        ['No contest', 'Draw', 'KO/TKO', 'Submission', 'Decision'], default='Other')
    for label, col in [('KO/TKO', 'is_ko'), ('Submission', 'is_submission'), ('Decision', 'is_decision')]:
        fights[col] = fights.outcome_group.eq(label)
    fights['is_finish'] = fights.is_ko | fights.is_submission
    fights['era'] = pd.cut(fights.fight_year, [1992, 1999, 2009, 2019, 2026],
                          labels=['1993–1999', '2000–2009', '2010–2019', '2020–2026']).astype('string')
    attrs = ['fighter_id', 'fighter_name', 'height_inches', 'weight_lbs', 'reach_inches', 'stance', 'dob']
    for corner in ('r', 'b'):
        dimension = fighters[attrs].rename(columns={c: f'{corner}_{c}' for c in attrs if c != 'fighter_id'})
        fights = fights.merge(dimension.rename(columns={'fighter_id': f'{corner}_id'}), on=f'{corner}_id', how='left', validate='many_to_one')
        ages = (fights.event_date - fights[f'{corner}_dob']).dt.days / 365.2425
        bad = ages.notna() & ~ages.between(16, 65)
        report(f'Implausible age in {corner} corner', bad.sum(), 'Set to missing; preserve original DOB', 'warning')
        fights[f'{corner}_age'] = ages.mask(bad)
    for col in ['age', 'height_inches', 'reach_inches', 'weight_lbs']:
        fights[f'{col}_difference'] = fights[f'r_{col}'] - fights[f'b_{col}']

    rounds = raw['round'].copy()
    if (~rounds.fight_id.isin(fights.fight_id)).any():
        raise ValueError('Orphan round fight IDs')
    meta = fights[['fight_id', 'r_id', 'b_id', 'finish_round', 'finish_time', 'time_format', 'fight_duration_seconds',
                   'in_scope', 'fight_year', 'event_date', 'weight_class', 'winner_id', 'decisive']]
    rounds = rounds.merge(meta, on='fight_id', suffixes=('', '_fight'), validate='many_to_one')
    bad_corners = rounds.r_id.ne(rounds.r_id_fight) | rounds.b_id.ne(rounds.b_id_fight)
    if bad_corners.any():
        raise ValueError('Round corners disagree with fight corners')
    rounds['round_duration_seconds'] = rounds.apply(
        lambda r: parse_clock(r.finish_time) if r.round_no == r.finish_round else
        ((round_schedule(r.time_format, r.finish_round) or [np.nan]*int(r.finish_round))[int(r.round_no)-1]
         if r.round_no <= r.finish_round else np.nan), axis=1)
    rounds.loc[rounds.fight_duration_seconds.isna(), 'round_duration_seconds'] = np.nan
    long_parts = []
    for corner, other in [('r', 'b'), ('b', 'r')]:
        part = rounds[['fight_id', 'round_no', 'round_duration_seconds', 'in_scope', 'fight_year', 'event_date', 'weight_class', 'decisive']].copy()
        part['fighter_id'] = rounds[f'{corner}_id']
        part['opponent_id'] = rounds[f'{other}_id']
        part['won'] = rounds[f'{corner}_id'].eq(rounds.winner_id).fillna(False)
        for col in raw['round']:
            if col.startswith(corner+'_') and col != corner+'_id':
                name = col[2:]
                values = rounds[col].map(parse_clock) if name == 'ctrl' else pd.to_numeric(rounds[col], errors='coerce')
                if name == 'ctrl':
                    name = 'ctrl_seconds'
                bad = values.lt(0)
                report(f'Negative {corner}_{name}', bad.sum(), 'Set analytical measure to missing', 'warning')
                part[name] = values.mask(bad)
        for landed, attempted in [('sig_landed', 'sig_atmp'), ('total_str_landed', 'total_str_atmp'), ('td_success', 'td_atmp')] + [
            (f'sig_str_landed_{x}', f'sig_str_atmp_{x}') for x in ('head', 'body', 'leg', 'distance', 'clinch', 'ground')]:
            invalid = part[landed].gt(part[attempted])
            report(f'{corner} {landed} exceeds attempts', invalid.sum(), 'Mask both measures', 'warning')
            part.loc[invalid, [landed, attempted]] = np.nan
        invalid_ctrl = part.ctrl_seconds.gt(part.round_duration_seconds)
        report(f'{corner} control exceeds round duration', invalid_ctrl.sum(), 'Set analytical control to missing', 'warning')
        part.loc[invalid_ctrl, 'ctrl_seconds'] = np.nan
        long_parts.append(part)
    long_rounds = pd.concat(long_parts, ignore_index=True)
    coverage = raw['round'].groupby('fight_id').round_no.agg(['count', 'min', 'max'])
    fights = fights.merge(coverage.add_prefix('stats_round_'), left_on='fight_id', right_index=True, how='left', validate='one_to_one')
    fights['round_stats_complete'] = (fights.stats_round_count.eq(fights.finish_round)
                                     & fights.stats_round_min.eq(1) & fights.stats_round_max.eq(fights.finish_round))
    report('Fights without complete round statistics', (~fights.round_stats_complete).sum(), 'Retain outcomes; exclude incomplete totals from performance metrics')
    report('Scrape errors with missing round stats', raw['scrape_error'].entity_id.isin(fights.loc[~fights.round_stats_complete, 'fight_id']).sum(), 'Coverage is nonrandom; display coverage by year')
    report('Master zero-filled fights without round records', raw['master'].fight_id.isin(fights.loc[fights.stats_round_count.isna(), 'fight_id']).sum(), 'Rebuild totals from round source; missing is not zero')
    if set(raw['master'].fight_id) != set(fights.fight_id):
        raise ValueError('Master and fight IDs differ')
    master = raw['master'].set_index('fight_id').sort_index()
    check = fights.set_index('fight_id').sort_index()
    for master_col, clean_col in [('event_id','event_id'), ('winner_id','winner_id'), ('r_fighter_id','r_id'),
                                  ('b_fighter_id','b_id'), ('event_name','event_name'), ('method','method'),
                                  ('weight_class','weight_class_raw'), ('finish_round','finish_round')]:
        n = master[master_col].fillna('<missing>').astype(str).ne(check[clean_col].fillna('<missing>').astype(str)).sum()
        report(f'Master reconciliation: {master_col}', n, 'Normalized source tables are authoritative', 'warning')
    return dict(events=events, fighters=fighters, fights=fights, rounds=long_rounds,
                bonuses=raw['fighter_bonus'].copy(), quality=pd.DataFrame(issues))
