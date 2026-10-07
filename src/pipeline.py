"""Run with python -m src.pipeline; all paths are repository-relative."""

import argparse

from .cleaning import clean_tables
from .data_loader import PROCESSED, TABLES, load_raw, profile_sources, verify_raw
from .feature_engineering import build_features


def prepare(save=True):
    verify_raw()
    raw = load_raw()
    if save:
        profile_sources(raw)
    data = build_features(clean_tables(raw))
    if save:
        from .validation import audit_snapshot

        audit_snapshot(data, raw)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        for name, frame in data.items():
            if name == "quality":
                frame.to_csv(TABLES / "quality_checks.csv", index=False)
            else:
                frame.to_pickle(PROCESSED / f"{name}.pkl")
        data["fights"].to_csv(PROCESSED / "fights.csv", index=False)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Also rebuild analyses, charts and models"
    )
    args = parser.parse_args()
    data = prepare()
    print(f"Prepared {data['fights'].in_scope.sum():,} scoped UFC fights")
    if args.all:
        from .analysis import export_analysis
        from .modeling import train_models
        from .statistics import export_statistics
        from .visualization import export_charts

        export_analysis(data)
        export_statistics(data)
        train_models(data["fights"])
        export_charts(data)
        from .reporting import build_findings

        build_findings()
        from .research import CHAPTERS, run_chapter
        from .research_reporting import build_research_report
        from .research_visualization import export_research_charts

        for number in sorted(CHAPTERS):
            run_chapter(number, data)
            print(f"Completed research chapter {number}", flush=True)
        export_research_charts()
        build_research_report()
        from .provenance import write_manifest

        write_manifest()
        print("Recorded analysis inputs and generated artifact hashes", flush=True)


if __name__ == "__main__":
    main()
