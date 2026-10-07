"""Render the research question catalog from computed evidence."""

import json

from .data_loader import ROOT, TABLES
from .research_notebooks import SECTIONS, notebook_for


def build_research_report():
    chapters = {
        number: json.loads(
            (TABLES / f"research_{number}.json").read_text(encoding="utf-8")
        )
        for number in sorted(SECTIONS)
    }
    count = sum(map(len, chapters.values()))
    lines = [
        "# UFC research questions and answers",
        "",
        f"{count} additional questions extend the eight original notebooks. Each answer is calculated from the preserved local snapshot and links to its supporting evidence. Inactivity research is included in notebook 04, bonuses in notebook 06, and Elo/generalization in notebook 08.",
        "",
        "These are exploratory observational analyses. Full 2025 is used for annual comparisons; 2026 is partial where retained. Wilson intervals describe binomial uncertainty without fighter/event clustering. Event-bootstrap intervals preserve dependence within events, but not recurring fighters across events. Intervals are pointwise and do not correct for selection across this question catalog.",
        "",
        "## Browse the research",
        "",
    ]
    for number in chapters:
        lines.append(f"- [{number} · {SECTIONS[number][0]}](#research-{number})")
    for number, answers in chapters.items():
        notebook_number = notebook_for(number)
        path = next((ROOT / "notebooks").glob(f"{notebook_number}_*.ipynb"))
        lines += [
            "",
            f'<a id="research-{number}"></a>',
            "",
            f"## {number} · {SECTIONS[number][0]}",
            "",
            f"[Executed notebook](notebooks/{path.name}) · [Answer data](outputs/tables/research_{number}.json)",
            "",
            f"![{SECTIONS[number][0]}](outputs/charts/research_{number}.png)",
            "",
        ]
        for item in answers:
            lines += [
                f"### {item['question']}",
                "",
                item["answer"],
                "",
                f"**Interpretation and limits:** {item['interpretation']}",
                "",
                f"[Supporting table](outputs/tables/{item['table']}.csv)",
                "",
            ]
    lines += [
        "## Reproduce the analysis",
        "",
        "From the repository root, install the pinned requirements and run:",
        "",
        "```powershell",
        "python -m pip install -r requirements.txt",
        "python -m src.notebooks --build --execute",
        "python -m src.pipeline --all",
        "python -m src.provenance",
        "python -m pytest -q",
        "```",
        "",
        "The full pipeline rebuilds all research tables, answer files, figures and this report. To refresh one notebook, run `python -m src.research_notebooks 04` (substitute 01–08). No downloads or API credentials are required for the analysis.",
        "",
        "The final pipeline run records SHA-256 digests of analysis source files, raw inputs and generated artifacts in `outputs/artifact_manifest.json`, along with Python/package versions. Verification detects changed, missing or newly added files; text line endings are normalized for Windows/Linux portability, while original raw CSVs remain byte-exact. It verifies consistency with the recorded full rebuild, not external data completeness or mathematical correctness. Individual notebook runs can change generated outputs; rerun the full pipeline before final verification. Notebook execution is checked separately.",
        "",
    ]
    path = ROOT / "RESEARCH.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    build_research_report()
