from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'dashboard'/'app.py'

@pytest.fixture(scope='module')
def app():
    return AppTest.from_file(str(APP),default_timeout=90).run()

@pytest.mark.parametrize('page',['Overview','UFC Evolution','Fighters','Fight Outcomes','Weight Classes',
    'Striking & Grappling','Events','Bonuses','Fighter Comparison','Fight Prediction / ML','Key Insights'])
def test_navigation(app,page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.message for e in app.exception]

def test_empty_filters_are_handled(app):
    app.sidebar.radio[0].set_value('Overview').run()
    app.sidebar.slider[0].set_value((1994,1994))
    app.sidebar.multiselect[0].set_value(["Women's Strawweight"]).run()
    assert not app.exception
    assert any('No fights match' in x.value for x in app.info)
    app.sidebar.multiselect[0].set_value([])
    app.sidebar.slider[0].set_value((1994,2026)).run()

def test_overview_denominator_reconciliation(app):
    app.sidebar.radio[0].set_value('Overview').run()
    metrics={x.label:x.value for x in app.metric}
    assert metrics['Fights']=='8,820'
    assert metrics['Events']=='783'
