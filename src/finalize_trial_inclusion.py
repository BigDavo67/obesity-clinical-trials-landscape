from pathlib import Path
import pandas as pd


# ============================================================
# FINALIZE TRIAL-LEVEL INCLUSION AND FILTER RELATED TABLES
# ============================================================

processed_data_dir = Path("data/processed")

triage_file = processed_data_dir / "trial_inclusion_triage.csv"
review_decisions_file = processed_data_dir / "review_decisions_final.csv"

interventions_file = processed_data_dir / "candidate_interventions.csv"
locations_file = processed_data_dir / "candidate_locations.csv"
collaborators_file = processed_data_dir / "candidate_collaborators.csv"


# ============================================================
# 1. LOAD FULL 1,548-TRIAL TRIAGE + REVIEWED DECISIONS
# ============================================================

triage_df = pd.read_csv(triage_file)
review_df = pd.read_csv(review_decisions_file)

print("=== Final inclusion merge ===")
print("Scoped triage trials:", len(triage_df))
print("Reviewed decision rows:", len(review_df))

if len(triage_df) != 1548:
    raise ValueError(
        f"Expected 1548 scoped trials in trial_inclusion_triage.csv, found {len(triage_df)}."
    )

if triage_df["nct_id"].duplicated().any():
    raise ValueError("Duplicate NCT IDs found in trial_inclusion_triage.csv.")

if review_df["nct_id"].duplicated().any():
    raise ValueError("Duplicate NCT IDs found in review_decisions_final.csv.")

unknown_review_ids = set(review_df["nct_id"]) - set(triage_df["nct_id"])
if unknown_review_ids:
    raise ValueError(
        f"Reviewed decision file contains NCT IDs absent from triage: {sorted(unknown_review_ids)[:10]}"
    )


# ============================================================
# 2. APPLY HUMAN/AUDIT DECISIONS
# ============================================================

decision_map = review_df.set_index("nct_id")["final_decision"]
reason_map = review_df.set_index("nct_id")["decision_reason"]
review_type_map = review_df.set_index("nct_id")["review_type"]

triage_df["final_decision"] = triage_df["nct_id"].map(decision_map)
triage_df["final_decision_reason"] = triage_df["nct_id"].map(reason_map)
triage_df["final_decision_source"] = triage_df["nct_id"].map(review_type_map)


# ============================================================
# 3. DEFAULT UNAUDITED STRONG AUTOMATIC CLASSIFICATIONS
# ============================================================

category_col = (
    "auto_review_category"
    if "auto_review_category" in triage_df.columns
    else "review_category"
)

unresolved_mask = triage_df["final_decision"].isna()

auto_include_default_mask = (
    unresolved_mask
    & (triage_df[category_col] == "LIKELY_INCLUDE")
)

triage_df.loc[
    auto_include_default_mask,
    "final_decision"
] = "INCLUDE"

triage_df.loc[
    auto_include_default_mask,
    "final_decision_source"
] = "AUTO_STRONG_INCLUDE"

triage_df.loc[
    auto_include_default_mask,
    "final_decision_reason"
] = (
    "Automatic include retained without further audit because the trial "
    "had strong weight-outcome and/or same-programme obesity-development evidence."
)

# Fallback for any unaudited automatic excludes.
# In the current frozen snapshot all 98 were audited, so this should be zero.
unresolved_mask = triage_df["final_decision"].isna()

auto_exclude_default_mask = (
    unresolved_mask
    & (triage_df[category_col] == "LIKELY_EXCLUDE")
)

triage_df.loc[
    auto_exclude_default_mask,
    "final_decision"
] = "EXCLUDE"

triage_df.loc[
    auto_exclude_default_mask,
    "final_decision_source"
] = "AUTO_EXCLUDE"

triage_df.loc[
    auto_exclude_default_mask,
    "final_decision_reason"
] = "Automatic exclude retained."


# ============================================================
# 4. VALIDATE EVERY TRIAL HAS A FINAL DECISION
# ============================================================

missing_decisions = triage_df["final_decision"].isna().sum()

if missing_decisions:
    unresolved = triage_df.loc[
        triage_df["final_decision"].isna(),
        ["nct_id", category_col]
    ]

    raise ValueError(
        "Some trials still lack a final decision:\n"
        + unresolved.head(20).to_string(index=False)
    )

invalid_decisions = set(
    triage_df["final_decision"].dropna().unique()
) - {"INCLUDE", "EXCLUDE"}

if invalid_decisions:
    raise ValueError(
        f"Unexpected final decisions found: {invalid_decisions}"
    )

print()
print("Final trial decisions:")
print(triage_df["final_decision"].value_counts())

print()
print("Decision provenance:")
print(triage_df["final_decision_source"].value_counts())


# ============================================================
# 5. SAVE FINAL TRIAL DECISION TABLE + INCLUDED TRIAL TABLE
# ============================================================

trial_decisions_file = (
    processed_data_dir
    / "trial_decisions_final.csv"
)

triage_df.to_csv(
    trial_decisions_file,
    index=False
)

included_trials_df = triage_df[
    triage_df["final_decision"] == "INCLUDE"
].copy()

included_trial_ids = set(
    included_trials_df["nct_id"]
)

clean_trials_file = (
    processed_data_dir
    / "obesity_trials_clean.csv"
)

included_trials_df.to_csv(
    clean_trials_file,
    index=False
)


# ============================================================
# 6. FILTER RELATED TABLES BY FINAL INCLUDED NCT IDs
# ============================================================

interventions_df = pd.read_csv(interventions_file)
locations_df = pd.read_csv(locations_file)
collaborators_df = pd.read_csv(collaborators_file)

clean_interventions_df = (
    interventions_df[
        interventions_df["nct_id"].isin(included_trial_ids)
    ]
    .drop_duplicates()
    .copy()
)

clean_locations_df = (
    locations_df[
        locations_df["nct_id"].isin(included_trial_ids)
    ]
    .drop_duplicates()
    .copy()
)

clean_collaborators_df = (
    collaborators_df[
        collaborators_df["nct_id"].isin(included_trial_ids)
    ]
    .drop_duplicates()
    .copy()
)

clean_interventions_file = (
    processed_data_dir
    / "obesity_interventions_clean.csv"
)

clean_locations_file = (
    processed_data_dir
    / "obesity_locations_clean.csv"
)

clean_collaborators_file = (
    processed_data_dir
    / "obesity_collaborators_clean.csv"
)

clean_interventions_df.to_csv(
    clean_interventions_file,
    index=False
)

clean_locations_df.to_csv(
    clean_locations_file,
    index=False
)

clean_collaborators_df.to_csv(
    clean_collaborators_file,
    index=False
)


# ============================================================
# 7. FINAL RELATIONAL VALIDATION
# ============================================================

orphan_interventions = (
    set(clean_interventions_df["nct_id"])
    - included_trial_ids
)

orphan_locations = (
    set(clean_locations_df["nct_id"])
    - included_trial_ids
)

orphan_collaborators = (
    set(clean_collaborators_df["nct_id"])
    - included_trial_ids
)

if orphan_interventions or orphan_locations or orphan_collaborators:
    raise ValueError("Orphan related-table NCT IDs detected.")

included_without_interventions = (
    included_trial_ids
    - set(clean_interventions_df["nct_id"])
)

print()
print("=== Final cleaned dataset sizes ===")
print("Trials:", len(included_trials_df))
print("Interventions:", len(clean_interventions_df))
print("Locations:", len(clean_locations_df))
print("Collaborators:", len(clean_collaborators_df))

print()
print(
    "Included trials with no intervention rows:",
    len(included_without_interventions)
)

print("Orphan intervention IDs:", len(orphan_interventions))
print("Orphan location IDs:", len(orphan_locations))
print("Orphan collaborator IDs:", len(orphan_collaborators))

print()
print("Saved:")
print(trial_decisions_file)
print(clean_trials_file)
print(clean_interventions_file)
print(clean_locations_file)
print(clean_collaborators_file)
