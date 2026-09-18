from pathlib import Path
import pandas as pd
import numpy as np

processed = Path("data/processed")

trials_file = processed / "obesity_trials_clean.csv"
taxonomy_file = processed / "obesity_focal_therapy_taxonomy_final.csv"

pair_out = processed / "phase5_trial_therapy_master.csv"
trial_out = processed / "phase5_trial_master.csv"
qa_out = processed / "phase5_master_qa.csv"

print("=== Phase 5.1: build analysis master tables ===")

trials = pd.read_csv(trials_file)
taxonomy = pd.read_csv(taxonomy_file)

print("Input clean trials:", len(trials))
print("Input taxonomy pairs:", len(taxonomy))

# ------------------------------------------------------------
# 1. REQUIRED KEYS / FROZEN BASELINE
# ------------------------------------------------------------

for df_name, df in [
    ("trials", trials),
    ("taxonomy", taxonomy),
]:
    if "nct_id" not in df.columns:
        raise ValueError(
            f"{df_name} is missing required key column 'nct_id'."
        )

if trials["nct_id"].duplicated().any():
    dupes = (
        trials.loc[
            trials["nct_id"].duplicated(keep=False),
            "nct_id"
        ]
        .drop_duplicates()
        .tolist()
    )
    raise ValueError(
        f"obesity_trials_clean.csv contains duplicate NCT IDs: {dupes[:20]}"
    )

pair_key = ["nct_id", "canonical_name_final"]

missing_pair_key = [
    c for c in pair_key
    if c not in taxonomy.columns
]
if missing_pair_key:
    raise ValueError(
        f"Taxonomy is missing pair-key columns: {missing_pair_key}"
    )

if taxonomy.duplicated(subset=pair_key).any():
    dupes = taxonomy.loc[
        taxonomy.duplicated(
            subset=pair_key,
            keep=False
        ),
        pair_key
    ]
    raise ValueError(
        "Duplicate trial × focal-therapy pairs in taxonomy:\n"
        + dupes.head(30).to_string(index=False)
    )

EXPECTED_TRIALS = 986
EXPECTED_PAIRS = 1186

if len(trials) != EXPECTED_TRIALS:
    raise ValueError(
        f"Expected {EXPECTED_TRIALS} clean trials; found {len(trials)}."
    )

if len(taxonomy) != EXPECTED_PAIRS:
    raise ValueError(
        f"Expected {EXPECTED_PAIRS} taxonomy pairs; found {len(taxonomy)}."
    )

if taxonomy["nct_id"].nunique() != EXPECTED_TRIALS:
    raise ValueError(
        "Taxonomy does not represent exactly 986 unique trials."
    )

# ------------------------------------------------------------
# 2. PARSE TRIAL-LEVEL DATES
# ------------------------------------------------------------

date_cols = [
    "start_date",
    "primary_completion_date",
    "completion_date",
    "first_posted",
    "last_update",
]

for col in date_cols:
    if col in trials.columns:
        trials[col] = pd.to_datetime(
            trials[col],
            errors="coerce"
        )

if "start_date" not in trials.columns:
    raise ValueError(
        "Clean trial table is missing 'start_date'."
    )

trials["start_year"] = trials["start_date"].dt.year
trials["start_month"] = trials["start_date"].dt.month

# 2026 is a partial year in the frozen project snapshot.
trials["year_display"] = (
    trials["start_year"]
    .astype("Int64")
    .astype(str)
)
trials.loc[
    trials["start_year"].eq(2026),
    "year_display"
] = "2026 YTD"

# ------------------------------------------------------------
# 3. NORMALISE PHASE
# ------------------------------------------------------------

def normalise_phase(value):
    if pd.isna(value):
        return "Not applicable / not reported"

    s = str(value).upper().strip()

    # Order matters for combined phases.
    if "EARLY_PHASE1" in s or "EARLY PHASE 1" in s:
        return "Early Phase 1"

    if (
        ("PHASE1" in s or "PHASE 1" in s)
        and ("PHASE2" in s or "PHASE 2" in s)
    ):
        return "Phase 1/2"

    if (
        ("PHASE2" in s or "PHASE 2" in s)
        and ("PHASE3" in s or "PHASE 3" in s)
    ):
        return "Phase 2/3"

    if "PHASE4" in s or "PHASE 4" in s:
        return "Phase 4"

    if "PHASE3" in s or "PHASE 3" in s:
        return "Phase 3"

    if "PHASE2" in s or "PHASE 2" in s:
        return "Phase 2"

    if "PHASE1" in s or "PHASE 1" in s:
        return "Phase 1"

    if "NA" == s or "N/A" in s or "NOT APPLICABLE" in s:
        return "Not applicable / not reported"

    return str(value)

if "phase" in trials.columns:
    trials["phase_group"] = trials["phase"].map(
        normalise_phase
    )
else:
    trials["phase_group"] = "Not applicable / not reported"

phase_order_map = {
    "Early Phase 1": 1,
    "Phase 1": 2,
    "Phase 1/2": 3,
    "Phase 2": 4,
    "Phase 2/3": 5,
    "Phase 3": 6,
    "Phase 4": 7,
    "Not applicable / not reported": 8,
}

trials["phase_order"] = (
    trials["phase_group"]
    .map(phase_order_map)
    .fillna(99)
    .astype(int)
)

# ------------------------------------------------------------
# 4. NORMALISE STATUS
# ------------------------------------------------------------

def normalise_status(value):
    if pd.isna(value):
        return "Unknown"

    s = str(value).upper().strip()

    active = {
        "RECRUITING",
        "NOT_YET_RECRUITING",
        "ACTIVE_NOT_RECRUITING",
        "ENROLLING_BY_INVITATION",
    }

    stopped = {
        "TERMINATED",
        "WITHDRAWN",
        "SUSPENDED",
    }

    if s in active:
        return "Active / recruiting"

    if s == "COMPLETED":
        return "Completed"

    if s in stopped:
        return "Stopped early"

    if s in {"UNKNOWN", "UNKNOWN_STATUS"}:
        return "Unknown"

    return "Other"

if "overall_status" in trials.columns:
    trials["status_group"] = trials[
        "overall_status"
    ].map(normalise_status)
else:
    trials["status_group"] = "Unknown"

# ------------------------------------------------------------
# 5. SIMPLE SPONSOR FIELDS
# ------------------------------------------------------------

if "lead_sponsor" in trials.columns:
    trials["lead_sponsor_clean"] = (
        trials["lead_sponsor"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
else:
    trials["lead_sponsor_clean"] = "Unknown"

if "lead_sponsor_class" in trials.columns:
    trials["sponsor_class_clean"] = (
        trials["lead_sponsor_class"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
elif "sponsor_class" in trials.columns:
    trials["sponsor_class_clean"] = (
        trials["sponsor_class"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )
else:
    trials["sponsor_class_clean"] = "Unknown"

# ------------------------------------------------------------
# 6. TRIAL-LEVEL FOCAL THERAPY COUNTS
# ------------------------------------------------------------

focal_summary = (
    taxonomy.groupby("nct_id")
    .agg(
        focal_therapy_pair_count=(
            "canonical_name_final",
            "size"
        ),
        focal_source_label_count=(
            "canonical_name_final",
            "nunique"
        ),
        focal_identity_count=(
            "therapy_identity_final",
            "nunique"
        ),
        verified_mechanism_pair_count=(
            "mechanism_classified",
            "sum"
        ),
    )
    .reset_index()
)

focal_summary["multi_focal_trial"] = (
    focal_summary["focal_identity_count"] > 1
)

trial_master = trials.merge(
    focal_summary,
    on="nct_id",
    how="left",
    validate="one_to_one",
)

if trial_master[
    "focal_therapy_pair_count"
].isna().any():
    missing_ids = trial_master.loc[
        trial_master[
            "focal_therapy_pair_count"
        ].isna(),
        "nct_id"
    ].tolist()
    raise ValueError(
        "Trials missing from focal taxonomy: "
        + str(missing_ids[:20])
    )

# ------------------------------------------------------------
# 7. BUILD PAIR-LEVEL MASTER
# ------------------------------------------------------------

# Avoid duplicate metadata columns already carried in taxonomy:
# add only trial columns not already present, plus nct_id.
trial_cols_to_add = [
    c for c in trial_master.columns
    if c == "nct_id" or c not in taxonomy.columns
]

pair_master = taxonomy.merge(
    trial_master[trial_cols_to_add],
    on="nct_id",
    how="left",
    validate="many_to_one",
)

if len(pair_master) != EXPECTED_PAIRS:
    raise ValueError(
        "Pair master row count changed during merge: "
        f"expected {EXPECTED_PAIRS}, found {len(pair_master)}."
    )

if pair_master["nct_id"].nunique() != EXPECTED_TRIALS:
    raise ValueError(
        "Pair master does not represent exactly 986 trials."
    )

# ------------------------------------------------------------
# 8. ADD ANALYSIS FLAGS
# ------------------------------------------------------------

pair_master["analysis_year_in_scope"] = (
    pair_master["start_year"]
    .between(2015, 2026, inclusive="both")
)

pair_master["is_2026_ytd"] = (
    pair_master["start_year"].eq(2026)
)

pair_master["verified_for_mechanism_chart"] = (
    pair_master["classification_status_final"]
    .isin(["VERIFIED", "VERIFIED_ALIAS"])
)

pair_master["unresolved_public_info"] = (
    pair_master["classification_status_final"]
    .eq("UNRESOLVED_PUBLIC_INFO")
)

pair_master["unclassified_long_tail"] = (
    pair_master["classification_status_final"]
    .eq("UNCLASSIFIED_LONG_TAIL")
)

# ------------------------------------------------------------
# 9. QA SUMMARY
# ------------------------------------------------------------

qa_rows = [
    {
        "check": "trial_master_rows",
        "value": len(trial_master),
        "expected": EXPECTED_TRIALS,
    },
    {
        "check": "trial_master_unique_nct",
        "value": trial_master["nct_id"].nunique(),
        "expected": EXPECTED_TRIALS,
    },
    {
        "check": "pair_master_rows",
        "value": len(pair_master),
        "expected": EXPECTED_PAIRS,
    },
    {
        "check": "pair_master_unique_nct",
        "value": pair_master["nct_id"].nunique(),
        "expected": EXPECTED_TRIALS,
    },
    {
        "check": "verified_pair_count",
        "value": int(
            pair_master[
                "verified_for_mechanism_chart"
            ].sum()
        ),
        "expected": 972,
    },
    {
        "check": "unresolved_public_info_pairs",
        "value": int(
            pair_master[
                "unresolved_public_info"
            ].sum()
        ),
        "expected": 26,
    },
    {
        "check": "unclassified_long_tail_pairs",
        "value": int(
            pair_master[
                "unclassified_long_tail"
            ].sum()
        ),
        "expected": 188,
    },
    {
        "check": "identity_mapping_conflicts",
        "value": int(
            pair_master.get(
                "has_mapping_conflict",
                pd.Series(False, index=pair_master.index)
            ).fillna(False).astype(bool).sum()
        ),
        "expected": 0,
    },
]

qa = pd.DataFrame(qa_rows)

qa["passed"] = (
    qa["value"] == qa["expected"]
)

if not qa["passed"].all():
    print()
    print(qa.to_string(index=False))
    raise ValueError(
        "At least one Phase 5.1 QA check failed."
    )

# ------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------

trial_master.to_csv(
    trial_out,
    index=False
)

pair_master.to_csv(
    pair_out,
    index=False
)

qa.to_csv(
    qa_out,
    index=False
)

print()
print("=== Phase 5.1 validation ===")
print(qa.to_string(index=False))

print()
print("Trial start years:")
print(
    trial_master["start_year"]
    .value_counts(dropna=False)
    .sort_index()
)

print()
print("Phase groups:")
print(
    trial_master["phase_group"]
    .value_counts()
)

print()
print("Status groups:")
print(
    trial_master["status_group"]
    .value_counts()
)

print()
print(
    "Multi-focal trials:",
    int(trial_master["multi_focal_trial"].sum())
)

print()
print("Saved:")
print(trial_out)
print(pair_out)
print(qa_out)

print()
print("PHASE 5.1 MASTER TABLE BUILD COMPLETE.")
