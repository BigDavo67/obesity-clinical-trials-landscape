from pathlib import Path
import pandas as pd

processed = Path("data/processed")

clean_file = processed / "obesity_trials_clean.csv"
master_file = processed / "phase5_trial_master.csv"
out_file = processed / "phase5_missing_start_date_audit.csv"

clean = pd.read_csv(clean_file)
master = pd.read_csv(master_file)

print("=== Phase 5.2A: start-date audit ===")
print("Clean trials:", len(clean))
print("Master trials:", len(master))

# ------------------------------------------------------------
# 1. Show all start-related columns available in the clean file
# ------------------------------------------------------------

start_cols = [
    c for c in clean.columns
    if "start" in c.lower()
]

print()
print("Start-related columns in obesity_trials_clean.csv:")
print(start_cols)

# ------------------------------------------------------------
# 2. Re-parse the raw start_date robustly
# ------------------------------------------------------------

if "start_date" not in clean.columns:
    raise ValueError("obesity_trials_clean.csv has no start_date column.")

raw = clean["start_date"].copy()

parsed = pd.to_datetime(
    raw,
    errors="coerce"
)

missing_parse = parsed.isna()

print()
print("Raw start_date missing/unparseable:", int(missing_parse.sum()))
print("Raw start_date present:", int((~missing_parse).sum()))

# Distinguish genuinely blank from nonblank-but-unparseable.
raw_text = raw.fillna("").astype(str).str.strip()

blank_mask = (
    raw.isna()
    | raw_text.eq("")
    | raw_text.str.lower().isin(["nan", "none", "nat"])
)

nonblank_unparseable = (
    missing_parse & ~blank_mask
)

print("Genuinely blank start_date:", int(blank_mask.sum()))
print(
    "Nonblank but unparseable start_date:",
    int(nonblank_unparseable.sum())
)

if nonblank_unparseable.any():
    print()
    print("Examples of nonblank/unparseable values:")
    print(
        clean.loc[
            nonblank_unparseable,
            ["nct_id", "start_date"]
        ]
        .head(30)
        .to_string(index=False)
    )

# ------------------------------------------------------------
# 3. Build audit table for trials with no usable start date
# ------------------------------------------------------------

candidate_cols = [
    "nct_id",
    "brief_title",
    "official_title",
    "lead_sponsor",
    "lead_sponsor_class",
    "phase",
    "overall_status",
    "start_date",
    "start_date_type",
    "primary_completion_date",
    "completion_date",
    "first_posted",
    "last_update",
    "enrollment",
    "enrollment_type",
]

audit_cols = [
    c for c in candidate_cols
    if c in clean.columns
]

audit = clean.loc[
    missing_parse,
    audit_cols
].copy()

# Add useful age-from-posting fields where available.
if "first_posted" in audit.columns:
    audit["first_posted_parsed"] = pd.to_datetime(
        audit["first_posted"],
        errors="coerce"
    )
    audit["first_posted_year"] = (
        audit["first_posted_parsed"].dt.year
    )

if "last_update" in audit.columns:
    audit["last_update_parsed"] = pd.to_datetime(
        audit["last_update"],
        errors="coerce"
    )

audit.to_csv(out_file, index=False)

# ------------------------------------------------------------
# 4. Summaries
# ------------------------------------------------------------

print()
print("=== Missing-start trials by status ===")
if "overall_status" in audit.columns:
    print(
        audit["overall_status"]
        .fillna("Unknown")
        .value_counts()
    )

print()
print("=== Missing-start trials by phase ===")
if "phase" in audit.columns:
    print(
        audit["phase"]
        .fillna("Unknown")
        .value_counts()
    )

print()
print("=== Missing-start trials by first-posted year ===")
if "first_posted_year" in audit.columns:
    print(
        audit["first_posted_year"]
        .value_counts(dropna=False)
        .sort_index()
    )

# ------------------------------------------------------------
# 5. Compare clean-file missing count with Phase 5 master
# ------------------------------------------------------------

master_start_missing = (
    master["start_year"].isna()
    if "start_year" in master.columns
    else pd.Series(False, index=master.index)
)

print()
print("=== Cross-check ===")
print(
    "Missing start year in phase5_trial_master:",
    int(master_start_missing.sum())
)

if int(master_start_missing.sum()) != int(missing_parse.sum()):
    raise ValueError(
        "Mismatch between clean-file missing dates and master missing years."
    )

# ------------------------------------------------------------
# 6. Check start_date_type if available
# ------------------------------------------------------------

if "start_date_type" in clean.columns:
    print()
    print("=== Start-date type across all trials ===")
    print(
        clean["start_date_type"]
        .fillna("Unknown")
        .value_counts()
    )

    usable = clean.loc[~missing_parse].copy()

    print()
    print("=== Start-date type among usable dates ===")
    print(
        usable["start_date_type"]
        .fillna("Unknown")
        .value_counts()
    )

print()
print("Saved audit file:")
print(out_file)
print()
print("PHASE 5.2A START-DATE AUDIT COMPLETE.")
