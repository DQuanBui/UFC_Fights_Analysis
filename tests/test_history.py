import pandas as pd
from src.feature_engineering import add_history

def sample():
    return pd.DataFrame({'fighter_id':['a']*4, 'event_date':pd.to_datetime(['2020-01-01','2020-02-01','2020-02-01','2020-03-01']),
        'is_ufc':[True]*4, 'won':[True,True,False,True], 'lost':[False,False,True,False],
        'ko_win':[True,False,False,False], 'submission_win':[False]*4, 'title_fight':[0]*4,
        'round_stats_complete':[True]*4,'fight_duration_seconds':[300]*4,
        'sig_landed':[10,20,30,40], 'sig_atmp':[20,30,40,50], 'td_success':[0]*4,'td_atmp':[0]*4})

def test_history_excludes_same_day_and_future():
    original=sample(); result=add_history(original)
    assert result.prior_ufc_fights.tolist()==[0,1,1,3]
    assert result.prior_wins.tolist()==[0,1,1,2]
    assert result.prior_win_streak.tolist()==[0,1,1,0]
    changed=original.copy(); changed.loc[3,'sig_landed']=9000
    changed.loc[3,'won']=False
    pd.testing.assert_frame_equal(result.filter(like='prior_'),add_history(changed).filter(like='prior_'))
