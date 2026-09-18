from pathlib import Path
import pandas as pd

processed = Path("data/processed")

manual_file = processed / "phase46_manual_role_decisions.csv"
scope_file = processed / "phase46_scope_corrections.csv"
role_candidates_file = processed / "phase46_intervention_role_candidates.csv"

trial_decisions_file = processed / "trial_decisions_final.csv"
review_decisions_file = processed / "review_decisions_final.csv"

trials_file = processed / "obesity_trials_clean.csv"
interventions_file = processed / "obesity_interventions_clean.csv"
locations_file = processed / "obesity_locations_clean.csv"
collaborators_file = processed / "obesity_collaborators_clean.csv"

manual = pd.read_csv(manual_file)
scope = pd.read_csv(scope_file)
roles = pd.read_csv(role_candidates_file)

print("=== Phase 4.6 finalization ===")
print("Manual role decisions:", len(manual))
print("Scope corrections:", len(scope))
print("Role candidate rows:", len(roles))

# ------------------------------------------------------------
# 1. APPLY 11 LATE SCOPE CORRECTIONS TO TRIAL DECISION FILES
# ------------------------------------------------------------

scope_ids = set(scope["nct_id"])

trial_decisions = pd.read_csv(trial_decisions_file)

trial_decisions.loc[
    trial_decisions["nct_id"].isin(scope_ids),
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(scope_ids),
    "final_decision_source"
] = "PHASE46_ROLE_SCOPE_CHECK"

reason_map = scope.set_index("nct_id")["decision_reason"]
trial_decisions.loc[
    trial_decisions["nct_id"].isin(scope_ids),
    "final_decision_reason"
] = trial_decisions.loc[
    trial_decisions["nct_id"].isin(scope_ids),
    "nct_id"
].map(reason_map)

trial_decisions.to_csv(trial_decisions_file, index=False)

# Patch review_decisions_final.csv so a future rerun remains reproducible.
review = pd.read_csv(review_decisions_file)

review = review[
    ~review["nct_id"].isin(scope_ids)
].copy()

scope_review = pd.DataFrame({
    "nct_id": scope["nct_id"],
    "review_type": "PHASE46_ROLE_SCOPE_CHECK",
    "initial_category": "POST_ROLE_AUDIT",
    "final_decision": "EXCLUDE",
    "decision_confidence": "high",
    "decision_reason": scope["decision_reason"],
    "source_url": scope["source_url"],
})

review = pd.concat([review, scope_review], ignore_index=True)

if review["nct_id"].duplicated().any():
    raise ValueError("Duplicate NCT IDs after patching review_decisions_final.csv.")

review.to_csv(review_decisions_file, index=False)

# ------------------------------------------------------------
# 2. FILTER CURRENT CLEAN RELATIONAL TABLES TO REVISED SCOPE
# ------------------------------------------------------------

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

trials = pd.read_csv(trials_file)
interventions = pd.read_csv(interventions_file)
locations = pd.read_csv(locations_file)
collaborators = pd.read_csv(collaborators_file)

trials = trials[trials["nct_id"].isin(included_ids)].copy()
interventions = interventions[interventions["nct_id"].isin(included_ids)].copy()
locations = locations[locations["nct_id"].isin(included_ids)].copy()
collaborators = collaborators[collaborators["nct_id"].isin(included_ids)].copy()

trials.to_csv(trials_file, index=False)
interventions.to_csv(interventions_file, index=False)
locations.to_csv(locations_file, index=False)
collaborators.to_csv(collaborators_file, index=False)

# ------------------------------------------------------------
# 3. APPLY MANUAL ROLE DECISIONS TO ALL 1,587 CANDIDATE ROWS
# ------------------------------------------------------------

key_cols = [
    "nct_id",
    "intervention_name",
    "active_component_source",
    "analysis_name_candidate",
    "arm_group_labels",
    "intervention_description",
]

def make_key(df):
    parts = []
    for col in key_cols:
        if col not in df.columns:
            raise ValueError(f"Missing matching column: {col}")
        parts.append(df[col].fillna("").astype(str).str.strip())
    key = parts[0]
    for part in parts[1:]:
        key = key + "|||" + part
    return key

roles["role_match_key"] = make_key(roles)

if roles["role_match_key"].duplicated().any():
    raise ValueError("Full role candidate table has duplicate role_match_key values.")

if manual["role_match_key"].duplicated().any():
    raise ValueError("Manual role decision file has duplicate role_match_key values.")

manual_map = manual.set_index("role_match_key")

roles["final_role"] = roles["role_match_key"].map(
    manual_map["final_role"]
)
roles["role_review_confidence_final"] = roles["role_match_key"].map(
    manual_map["role_review_confidence"]
)
roles["role_review_reason_final"] = roles["role_match_key"].map(
    manual_map["role_review_reason"]
)
roles["canonical_name_final"] = roles["role_match_key"].map(
    manual_map["canonical_name_final"]
)

# Rows not in the manual-review subset retain the prior high/medium-confidence result.
unreviewed = roles["final_role"].isna()

roles.loc[unreviewed, "final_role"] = roles.loc[
    unreviewed, "role_candidate"
]

roles.loc[unreviewed, "role_review_confidence_final"] = roles.loc[
    unreviewed, "role_confidence"
]

roles.loc[unreviewed, "role_review_reason_final"] = roles.loc[
    unreviewed, "role_reason"
]

roles.loc[unreviewed, "canonical_name_final"] = roles.loc[
    unreviewed, "analysis_name_candidate"
]

# Anything from newly excluded trials is explicitly out of scope.
roles.loc[
    roles["nct_id"].isin(scope_ids),
    "final_role"
] = "OUT_OF_SCOPE"

roles.loc[
    roles["nct_id"].isin(scope_ids),
    "role_review_reason_final"
] = roles.loc[
    roles["nct_id"].isin(scope_ids),
    "nct_id"
].map(reason_map)

# ------------------------------------------------------------
# 4. VALIDATE
# ------------------------------------------------------------

if roles["final_role"].isna().any():
    raise ValueError("Some role rows still lack a final role.")

valid_roles = {
    "FOCAL_OBESITY_THERAPY",
    "ACTIVE_COMPARATOR",
    "PK_DDI_PROBE",
    "POSITIVE_CONTROL",
    "BACKGROUND_OR_CONCOMITANT",
    "CONTROL_PLACEBO",
    "OUT_OF_SCOPE",
}

unknown_roles = set(roles["final_role"].unique()) - valid_roles
if unknown_roles:
    raise ValueError(f"Unexpected final roles: {unknown_roles}")

# ------------------------------------------------------------
# 5. SAVE FINAL ROLE TABLES
# ------------------------------------------------------------

all_roles_file = processed / "obesity_intervention_roles_final.csv"
roles.to_csv(all_roles_file, index=False)

included_roles = roles[
    roles["nct_id"].isin(included_ids)
    & ~roles["final_role"].isin(["CONTROL_PLACEBO", "OUT_OF_SCOPE"])
].copy()

analysis_roles_file = processed / "obesity_interventions_analysis_ready.csv"
included_roles.to_csv(analysis_roles_file, index=False)

focal = included_roles[
    included_roles["final_role"] == "FOCAL_OBESITY_THERAPY"
].copy()

# One row per trial × canonical focal therapy identity.
trial_therapy_map = (
    focal[
        [
            "nct_id",
            "canonical_name_final",
            "lead_sponsor",
            "lead_sponsor_class",
            "brief_title",
            "phase",
            "overall_status",
        ]
    ]
    .drop_duplicates(
        subset=["nct_id", "canonical_name_final"]
    )
    .sort_values(["nct_id", "canonical_name_final"])
)

trial_therapy_file = processed / "obesity_trial_focal_therapy_map.csv"
trial_therapy_map.to_csv(trial_therapy_file, index=False)

# ------------------------------------------------------------
# 6. FINAL CHECKS
# ------------------------------------------------------------

print()
print("Revised trial decisions:")
print(trial_decisions["final_decision"].value_counts())

print()
print("Final intervention roles:")
print(roles["final_role"].value_counts())

print()
print("=== Revised clean dataset sizes ===")
print("Trials:", len(trials))
print("Interventions:", len(interventions))
print("Locations:", len(locations))
print("Collaborators:", len(collaborators))

print()
print("Analysis-ready intervention rows:", len(included_roles))
print("Focal therapy rows:", len(focal))
print("Unique trial × focal-therapy pairs:", len(trial_therapy_map))
print("Trials represented in focal-therapy map:", trial_therapy_map["nct_id"].nunique())

missing_focal_trials = included_ids - set(trial_therapy_map["nct_id"])
print("Included trials with no focal therapy after final role classification:", len(missing_focal_trials))

if missing_focal_trials:
    print(sorted(missing_focal_trials)[:30])

print()
print("Saved:")
print(all_roles_file)
print(analysis_roles_file)
print(trial_therapy_file)
