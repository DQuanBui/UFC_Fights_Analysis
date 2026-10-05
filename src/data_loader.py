"""Load the pinned local snapshot and profile every source field."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
NAMES = (
    "event",
    "fight",
    "fighter",
    "fighter_bonus",
    "master",
    "round",
    "scrape_error",
)
KEYS = {
    "event": ["event_id"],
    "fight": ["fight_id"],
    "fighter": ["fighter_id"],
    "fighter_bonus": ["fight_id", "bonus_type"],
    "master": ["fight_id"],
    "round": ["fight_id", "round_no"],
    "scrape_error": ["error_id"],
}


def load_raw(raw_dir=RAW):
    """Read all seven tables; identifiers are strings, never numeric measures."""
    tables = {}
    for name in NAMES:
        path = Path(raw_dir) / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing source: {path}")
        frame = pd.read_csv(path)
        for col in frame:
            if col.endswith("_id"):
                frame[col] = frame[col].astype("string")
        tables[name] = frame
    return tables


def verify_raw(raw_dir=RAW):
    """Fail if a supplied source differs from its original SHA-256 digest."""
    folder = Path(raw_dir)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8-sig"))
    for item in manifest:
        actual = hashlib.sha256((folder / item["file"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Source checksum mismatch: {item['file']}")
    return len(manifest)


def profile_sources(tables, output_dir=TABLES):
    """Save field-level quality, sample rows, category counts and key checks."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fields, summary, categories, dates = [], [], [], []
    for name, frame in tables.items():
        summary.append(
            dict(
                table=name,
                rows=len(frame),
                columns=len(frame.columns),
                duplicate_rows=int(frame.duplicated().sum()),
                duplicate_keys=int(frame.duplicated(KEYS[name]).sum()),
                missing_keys=int(frame[KEYS[name]].isna().any(axis=1).sum()),
            )
        )
        for col in frame:
            s = frame[col]
            fields.append(
                dict(
                    table=name,
                    column=col,
                    dtype=str(s.dtype),
                    missing=int(s.isna().sum()),
                    missing_pct=100 * s.isna().mean(),
                    unique=s.nunique(),
                    examples=" | ".join(s.dropna().astype(str).unique()[:3]),
                )
            )
            if col in ("date", "event_date", "dob", "occurred_at"):
                parsed = pd.to_datetime(s, errors="coerce")
                dates.append(
                    dict(
                        table=name,
                        column=col,
                        minimum=str(parsed.min()),
                        maximum=str(parsed.max()),
                        malformed=int((s.notna() & parsed.isna()).sum()),
                    )
                )
            if s.nunique() <= 200 and not col.endswith("_id"):
                for value, count in s.fillna("<missing>").value_counts().items():
                    categories.append(
                        dict(table=name, column=col, value=value, count=count)
                    )
        frame.head(3).to_csv(output / f"sample_{name}.csv", index=False)
    results = {
        "source_summary": pd.DataFrame(summary),
        "data_dictionary": pd.DataFrame(fields),
        "category_counts": pd.DataFrame(categories),
        "date_coverage": pd.DataFrame(dates),
    }
    for name, frame in results.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    return results


if __name__ == "__main__":
    print(f"Verified {verify_raw()} source checksums")
    print(profile_sources(load_raw())["source_summary"].to_string(index=False))
