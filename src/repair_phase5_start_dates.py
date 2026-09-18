from pathlib import Path
import pandas as pd
import numpy as np

processed = Path("data/processed")

trial_file = processed / "phase5_trial_master.csv"
pair_file = processed / "phase5_trial_therapy_master.csv"
qa_file = processed / "phase5_start_date_precision_qa.csv"

trial = pd.read_csv(trial_file)
pair = pd.read_csv(pair_file)

SNAPSHOT_DATE = pd.Timestamp("2026-09-17")

print("=== Phase 5.2B: repair start-date parsing ===")
print("Trial rows:", len(trial))
print("Pair rows:", len(pair))

if "start_date" not in trial.columns:
    raise ValueError("phase5_trial_master.csv is missing start_date.")

def add_start_fields(df):
    s = (
        df["start_date"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ClinicalTrials.gov dates in this project may be:
    # YYYY
    # YYYY-MM
    # YYYY-MM-DD
    parts = s.str.extract(
        r"^(?P<year>\d{4})(?:-(?P<month>\d{2}))?(?:-(?P<day>\d{2}))?$"
    )

    df["start_year"] = pd.to_numeric(
        parts["year"], errors="coerce"
    ).astype("Int64")

    df["start_month"] = pd.to_numeric(
        parts["month"], errors="coerce"
    ).astype("Int64")

    df["start_day"] = pd.to_numeric(
        parts["day"], errors="coerce"
    ).astype("Int64")

    df["start_date_precision"] = np.select(
        [
            df["start_day"].notna(),
            df["start_month"].notna(),
            df["start_year"].notna(),
        ],
        [
            "DAY",
            "MONTH",
            "YEAR",
        ],
        default="UNPARSEABLE",
    )

    df["year_display"] = (
        df["start_year"]
        .astype("Int64")
        .astype(str)
    )

    df.loc[
        df["start_year"].eq(2026),
        "year_display"
    ] = "2026 YTD"

    # Preserve uncertainty rather than inventing a day.
    # For exact DAY dates we can determine whether the trial start
    # is before/after the snapshot.
    exact_dates = pd.to_datetime(
        {
            "year": df["start_year"],
            "month": df["start_month"],
            "day": df["start_day"],
        },
        errors="coerce",
    )

    df["start_date_exact_for_comparison"] = exact_dates

    def timing(row):
        y = row["start_year"]
        m = row["start_month"]
        d = row["start_day"]

        if pd.isna(y):
            return "UNPARSEABLE"

        y = int(y)

        if y < SNAPSHOT_DATE.year:
            return "BEFORE_SNAPSHOT"
        if y > SNAPSHOT_DATE.year:
            return "AFTER_SNAPSHOT"

        # Same year as snapshot.
        if pd.isna(m):
            return "SNAPSHOT_YEAR_UNKNOWN_MONTH"

        m = int(m)

        if m < SNAPSHOT_DATE.month:
            return "BEFORE_SNAPSHOT"
        if m > SNAPSHOT_DATE.month:
            return "AFTER_SNAPSHOT"

        # Same month.
        if pd.isna(d):
            return "SNAPSHOT_MONTH_MONTH_PRECISION"

        d = int(d)
        if d <= SNAPSHOT_DATE.day:
            return "BEFORE_SNAPSHOT"
        return "AFTER_SNAPSHOT"

    df["start_timing_vs_snapshot"] = df.apply(
        timing,
        axis=1,
    )

    # Conservative YTD flag:
    # True only when we can say the reported start is on/before snapshot.
    # Month-precision September 2026 records remain NA rather than guessed.
    df["started_by_snapshot_conservative"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="boolean",
    )

    df.loc[
        df["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT"),
        "started_by_snapshot_conservative"
    ] = True

    df.loc[
        df["start_timing_vs_snapshot"].eq("AFTER_SNAPSHOT"),
        "started_by_snapshot_conservative"
    ] = False

    return df

trial = add_start_fields(trial)

# Propagate repaired fields to pair table from trial table to guarantee consistency.
repair_cols = [
    "nct_id",
    "start_year",
    "start_month",
    "start_day",
    "start_date_precision",
    "year_display",
    "start_date_exact_for_comparison",
    "start_timing_vs_snapshot",
    "started_by_snapshot_conservative",
]

pair = pair.drop(
    columns=[
        c for c in repair_cols
        if c != "nct_id" and c in pair.columns
    ],
    errors="ignore",
)

pair = pair.merge(
    trial[repair_cols],
    on="nct_id",
    how="left",
    validate="many_to_one",
)

# Recreate scope flag from repaired year.
pair["analysis_year_in_scope"] = (
    pair["start_year"]
    .between(2015, 2026, inclusive="both")
)

pair["is_2026_ytd"] = pair["start_year"].eq(2026)

# QA
qa_rows = [
    {
        "check": "trial_rows",
        "value": len(trial),
        "expected": 986,
    },
    {
        "check": "pair_rows",
        "value": len(pair),
        "expected": 1186,
    },
    {
        "check": "trial_start_year_missing",
        "value": int(trial["start_year"].isna().sum()),
        "expected": 0,
    },
    {
        "check": "pair_start_year_missing",
        "value": int(pair["start_year"].isna().sum()),
        "expected": 0,
    },
]

qa = pd.DataFrame(qa_rows)
qa["passed"] = qa["value"].eq(qa["expected"])

print()
print("=== Repaired start-date precision ===")
print(
    trial["start_date_precision"]
    .value_counts(dropna=False)
)

print()
print("=== Timing versus 2026-09-17 snapshot ===")
print(
    trial["start_timing_vs_snapshot"]
    .value_counts(dropna=False)
)

print()
print("=== Repaired trial start years ===")
print(
    trial["start_year"]
    .value_counts(dropna=False)
    .sort_index()
)

print()
print("=== QA ===")
print(qa.to_string(index=False))

if not qa["passed"].all():
    raise ValueError(
        "At least one Phase 5.2B start-date repair QA check failed."
    )

trial.to_csv(trial_file, index=False)
pair.to_csv(pair_file, index=False)
qa.to_csv(qa_file, index=False)

print()
print("Updated:")
print(trial_file)
print(pair_file)
print("Saved:")
print(qa_file)

print()
print("PHASE 5.2B START-DATE REPAIR COMPLETE.")
