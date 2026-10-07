"""Maintain research sections while preserving the original notebook cells."""

import argparse

import nbformat

from .data_loader import ROOT

SECTIONS = {
    "12": (
        "How does the frozen model generalize across fighter familiarity?",
        [
            "deep_generalization_familiarity",
            "deep_generalization_experience",
            "deep_generalization_errors",
        ],
    ),
    "11": (
        "What changes after a long gap between UFC appearances?",
        [
            "deep_layoff_bands",
            "deep_layoff_matched",
            "deep_layoff_age",
            "deep_layoff_eras",
        ],
    ),
    "10": (
        "Does opponent strength improve pre-fight prediction?",
        ["deep_elo_summary", "deep_elo_folds", "deep_elo_gain", "deep_elo_calibration"],
    ),
    "09": (
        "How do bonus era and award category change the interpretation?",
        [
            "deep_bonus_eras",
            "deep_bonus_categories",
            "deep_bonus_division_standardization",
        ],
    ),
    "08": (
        "Which features help, when does the model fail, and how robust is its lift?",
        [
            "deep_feature_ablation_summary",
            "deep_feature_ablation_folds",
            "deep_model_years",
            "deep_model_confidence",
            "deep_paired_model_gain",
        ],
    ),
    "07": (
        "Which physical associations survive adjustment and sensitivity checks?",
        [
            "deep_adjusted_attributes",
            "deep_within_division_correlation",
            "deep_reach_age_caliper",
        ],
    ),
    "06": (
        "What changes within fights after accounting for exposure and survival?",
        [
            "deep_round_exposure",
            "deep_paired_round_pace",
            "deep_opening_round_leads",
            "deep_control_coverage",
        ],
    ),
    "05": (
        "Can composition, schedules and rematches explain outcomes?",
        [
            "deep_decision_decomposition",
            "deep_divided_decisions",
            "deep_title_schedule",
            "deep_rematches",
        ],
    ),
    "04": (
        "What do rankings, debut matchups and early careers really reveal?",
        [
            "deep_ranking_stability",
            "deep_debut_matchups",
            "deep_age_gap",
            "deep_career_continuation",
        ],
    ),
    "03": (
        "What drives growth, geographic spread and fighter turnover?",
        [
            "deep_growth_decomposition",
            "deep_geographic_concentration",
            "deep_entrant_flow",
            "deep_annual_return",
        ],
    ),
    "02": (
        "How sensitive are the results to cleaning and metric definitions?",
        [
            "deep_timing_sensitivity",
            "deep_cohort_sensitivity",
            "deep_rate_weighting",
            "deep_zero_attempts",
        ],
    ),
    "01": (
        "Where can the dataset mislead us?",
        [
            "deep_scope_bias",
            "deep_missingness",
            "deep_missing_selection",
            "deep_zero_bias",
        ],
    ),
}

EXTRA_CHAPTERS = {"04": ["11"], "06": ["09"], "08": ["10", "12"]}


def notebook_for(number):
    return next(
        (book for book, extras in EXTRA_CHAPTERS.items() if number in extras), number
    )


def cells_for(number):
    title, tables = SECTIONS[number]
    cells = [
        nbformat.v4.new_markdown_cell(
            f"## Research extension: {title}\n\nThe following answers are computed from the current snapshot. Tables expose denominators and missingness so the claims can be checked."
        )
    ]
    cells.append(
        nbformat.v4.new_code_cell(
            f"from src.research import run_chapter, display_answers\ndeep_tables, deep_answers = run_chapter('{number}')\ndisplay_answers(deep_answers)"
        )
    )
    cells.append(
        nbformat.v4.new_code_cell(
            f"from src.research_visualization import research_chart\nshow(research_chart('{number}'))"
        )
    )
    for name in tables:
        cells.append(nbformat.v4.new_code_cell(f"display(deep_tables['{name}'])"))
    for cell in cells:
        cell.metadata["tags"] = ["research-extension"]
    for extra in EXTRA_CHAPTERS.get(number, []):
        cells.extend(cells_for(extra))
    return cells


def update(number, execute=True):
    number = notebook_for(number)
    path = next((ROOT / "notebooks").glob(number + "_*.ipynb"))
    notebook = nbformat.read(path, 4)
    notebook.cells = [
        c
        for c in notebook.cells
        if "research-extension" not in c.metadata.get("tags", [])
    ]
    notebook.cells.extend(cells_for(number))
    nbformat.write(notebook, path)
    if execute:
        from .notebooks import execute_notebooks

        execute_notebooks([path])
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", choices=sorted(SECTIONS))
    parser.add_argument("--no-execute", action="store_true")
    args = parser.parse_args()
    update(args.chapter, not args.no_execute)
