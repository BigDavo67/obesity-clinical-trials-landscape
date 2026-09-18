from pathlib import Path
import pandas as pd
import numpy as np

processed = Path("data/processed")

clean_file = processed / "obesity_trials_clean.csv"
trial_file = processed / "phase5_trial_master.csv"
pair_file = processed / "phase5_trial_therapy_master.csv"
qa_file = processed / "phase5_start_date_precision_qa.csv"

SNAPSHOT_DATE = pd.Timestamp("2026-09-17")

clean = pd.read_csv(clean_file, dtype={"start_date": "string"})
trial = pd.read_csv(trial_file)
pair = pd.read_csv(pair_file)

print("=== Phase 5.2B v3: restore raw start dates and repair parsing ===")
print("Clean trial rows:", len(clean))
print("Trial master rows:", len(trial))
print("Pair master rows:", len(pair))

# ------------------------------------------------------------
# 1. Structural checks
# ------------------------------------------------------------

if "start_date" not in clean.columns:
    raise ValueError(
        "obesity_trials_clean.csv is missing raw start_date."
    )

if clean["nct_id"].duplicated().any():
    raise ValueError(
        "obesity_trials_clean.csv contains duplicate NCT IDs."
    )

if len(clean) != 986:
    raise ValueError(
        f"Expected 986 clean trials; found {len(clean)}."
    )

if len(trial) != 986:
    raise ValueError(
        f"Expected 986 trial-master rows; found {len(trial)}."
    )

if len(pair) != 1186:
    raise ValueError(
        f"Expected 1186 pair-master rows; found {len(pair)}."
    )

# ------------------------------------------------------------
# 2. Restore the untouched ClinicalTrials.gov start-date string
# ------------------------------------------------------------

raw_dates = clean[
    ["nct_id", "start_date"]
].rename(
    columns={"start_date": "start_date_raw"}
)

trial = trial.drop(
    columns=[
        c for c in [
            "start_date_raw",
            "start_year",
            "start_month",
            "start_day",
            "start_date_precision",
            "year_display",
            "start_date_exact_for_comparison",
            "start_timing_vs_snapshot",
            "started_by_snapshot_conservative",
        ]
        if c in trial.columns
    ],
    errors="ignore",
)

trial = trial.merge(
    raw_dates,
    on="nct_id",
    how="left",
    validate="one_to_one",
)

# Keep start_date itself as the raw registry value from here onward.
trial["start_date"] = trial["start_date_raw"]

s = (
    trial["start_date_raw"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# ------------------------------------------------------------
# 3. Parse variable date precision explicitly
# ------------------------------------------------------------

parts = s.str.extract(
    r"^(?P<year>\d{4})(?:-(?P<month>\d{2}))?(?:-(?P<day>\d{2}))?$"
)

trial["start_year"] = pd.to_numeric(
    parts["year"], errors="coerce"
).astype("Int64")

trial["start_month"] = pd.to_numeric(
    parts["month"], errors="coerce"
).astype("Int64")

trial["start_day"] = pd.to_numeric(
    parts["day"], errors="coerce"
).astype("Int64")

trial["start_date_precision"] = np.select(
    [
        trial["start_day"].notna(),
        trial["start_month"].notna(),
        trial["start_year"].notna(),
    ],
    [
        "DAY",
        "MONTH",
        "YEAR",
    ],
    default="UNPARSEABLE",
)

trial["year_display"] = (
    trial["start_year"]
    .astype("Int64")
    .astype(str)
)

trial.loc[
    trial["start_year"].eq(2026),
    "year_display"
] = "2026 YTD"

# ------------------------------------------------------------
# 4. Exact Timestamp only where a full day is genuinely reported
# ------------------------------------------------------------

trial["start_date_exact_for_comparison"] = pd.NaT

full_date_mask = (
    trial["start_year"].notna()
    & trial["start_month"].notna()
    & trial["start_day"].notna()
)

if full_date_mask.any():
    exact_strings = (
        trial.loc[full_date_mask, "start_year"]
        .astype(int).astype(str).str.zfill(4)
        + "-"
        + trial.loc[full_date_mask, "start_month"]
        .astype(int).astype(str).str.zfill(2)
        + "-"
        + trial.loc[full_date_mask, "start_day"]
        .astype(int).astype(str).str.zfill(2)
    )

    trial.loc[
        full_date_mask,
        "start_date_exact_for_comparison"
    ] = pd.to_datetime(
        exact_strings,
        format="%Y-%m-%d",
        errors="coerce",
    ).values

# ------------------------------------------------------------
# 5. Timing relative to the frozen 17 Sep 2026 snapshot
#    without inventing a day for month-precision records
# ------------------------------------------------------------

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

    # Same month (September 2026).
    if pd.isna(d):
        return "SNAPSHOT_MONTH_MONTH_PRECISION"

    d = int(d)

    if d <= SNAPSHOT_DATE.day:
        return "BEFORE_SNAPSHOT"

    return "AFTER_SNAPSHOT"

trial["start_timing_vs_snapshot"] = trial.apply(
    timing,
    axis=1,
)

trial["started_by_snapshot_conservative"] = pd.Series(
    pd.NA,
    index=trial.index,
    dtype="boolean",
)

trial.loc[
    trial["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT"),
    "started_by_snapshot_conservative"
] = True

trial.loc[
    trial["start_timing_vs_snapshot"].eq("AFTER_SNAPSHOT"),
    "started_by_snapshot_conservative"
] = False

# ------------------------------------------------------------
# 6. Propagate repaired fields to pair master
# ------------------------------------------------------------

repair_cols = [
    "nct_id",
    "start_date",
    "start_date_raw",
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

pair["analysis_year_in_scope"] = (
    pair["start_year"]
    .between(2015, 2026, inclusive="both")
)

pair["is_2026_ytd"] = pair["start_year"].eq(2026)

# ------------------------------------------------------------
# 7. QA
# ------------------------------------------------------------

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
    {
        "check": "unparseable_trial_start_dates",
        "value": int(
            trial["start_date_precision"]
            .eq("UNPARSEABLE")
            .sum()
        ),
        "expected": 0,
    },
]

qa = pd.DataFrame(qa_rows)
qa["passed"] = qa["value"].eq(qa["expected"])

print()
print("=== Restored start-date precision ===")
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
print("=== 2026 timing detail ===")
print(
    trial.loc[
        trial["start_year"].eq(2026),
        "start_timing_vs_snapshot"
    ]
    .value_counts(dropna=False)
)

print()
print("=== QA ===")
print(qa.to_string(index=False))

if not qa["passed"].all():
    print()
    print("Unparseable examples:")
    print(
        trial.loc[
            trial["start_year"].isna(),
            ["nct_id", "start_date_raw"]
        ]
        .head(30)
        .to_string(index=False)
    )
    raise ValueError(
        "At least one Phase 5.2B v3 QA check failed."
    )

# ------------------------------------------------------------
# 8. Save only after all QA passes
# ------------------------------------------------------------

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
print("PHASE 5.2B V3 START-DATE REPAIR COMPLETE.")
