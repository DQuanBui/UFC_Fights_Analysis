from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP), default_timeout=90).run()


@pytest.mark.parametrize(
    "page",
    [
        "Overview",
        "UFC Evolution",
        "Fighters",
        "Fight Outcomes",
        "Weight Classes",
        "Striking & Grappling",
        "Events",
        "Bonuses",
        "Fighter Comparison",
        "Fight Prediction / ML",
        "Key Insights",
        "Research Questions",
    ],
)
def test_navigation(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.message for e in app.exception]


def test_empty_filters_are_handled(app):
    app.sidebar.radio[0].set_value("Overview").run()
    app.sidebar.slider[0].set_value((1994, 1994))
    app.sidebar.multiselect[0].set_value(["Women's Strawweight"]).run()
    assert not app.exception
    assert any("No fights match" in x.value for x in app.info)
    app.sidebar.multiselect[0].set_value([])
    app.sidebar.slider[0].set_value((1994, 2026)).run()


def test_overview_denominator_reconciliation(app):
    app.sidebar.radio[0].set_value("Overview").run()
    metrics = {x.label: x.value for x in app.metric}
    assert metrics["Fights"] == "8,820"
    assert metrics["Events"] == "783"


def test_partial_year_only_and_research_drilldown(app):
    app.sidebar.radio[0].set_value("Overview").run()
    app.sidebar.slider[0].set_value((2026, 2026)).run()
    assert not app.exception
    assert {x.label: x.value for x in app.metric}["Fights"] == "326"
    app.sidebar.radio[0].set_value("UFC Evolution").run()
    assert not app.exception
    app.sidebar.slider[0].set_value((1994, 2026)).run()


@pytest.mark.parametrize("number", [f"{n:02}" for n in range(1, 10)])
def test_research_evidence_topics(app, number):
    import json

    app.sidebar.radio[0].set_value("Research Questions").run()
    app.selectbox[0].set_value(number).run()
    assert not app.exception, [e.message for e in app.exception]
    answers = json.loads(
        (APP.parents[1] / "outputs" / "tables" / f"research_{number}.json").read_text(
            encoding="utf-8"
        )
    )
    assert [x.value for x in app.subheader] == [x["question"] for x in answers]
    assert len(app.dataframe) == len(answers)
    assert len(app.sidebar.slider) == 0
