from pathlib import Path
import pandas as pd

processed = Path("data/processed")

roles_file = processed / "obesity_intervention_roles_final.csv"
analysis_ready_file = processed / "obesity_interventions_analysis_ready.csv"
focal_map_file = processed / "obesity_trial_focal_therapy_map.csv"

trial_decisions_file = processed / "trial_decisions_final.csv"
review_decisions_file = processed / "review_decisions_final.csv"

trials_file = processed / "obesity_trials_clean.csv"
interventions_file = processed / "obesity_interventions_clean.csv"
locations_file = processed / "obesity_locations_clean.csv"
collaborators_file = processed / "obesity_collaborators_clean.csv"

nct_id = "NCT04557267"

reason = (
    "SKF7 is a Labisia pumila botanical-extract programme. "
    "This falls outside the frozen conventional pharmacotherapy scope, "
    "which excludes nutraceutical, traditional and food-derived interventions."
)

roles = pd.read_csv(roles_file)
trial_decisions = pd.read_csv(trial_decisions_file)
review_decisions = pd.read_csv(review_decisions_file)

trials = pd.read_csv(trials_file)
interventions = pd.read_csv(interventions_file)
locations = pd.read_csv(locations_file)
collaborators = pd.read_csv(collaborators_file)

print("=== Phase 4.7 SKF7 botanical scope correction ===")

if nct_id not in set(trial_decisions["nct_id"]):
    raise ValueError(f"{nct_id} not found in trial_decisions_final.csv.")

trial_decisions.loc[
    trial_decisions["nct_id"] == nct_id,
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"] == nct_id,
    "final_decision_source"
] = "PHASE47_BOTANICAL_SCOPE_CHECK"

trial_decisions.loc[
    trial_decisions["nct_id"] == nct_id,
    "final_decision_reason"
] = reason

trial_decisions.to_csv(trial_decisions_file, index=False)

review_decisions = review_decisions[
    review_decisions["nct_id"] != nct_id
].copy()

new_review = pd.DataFrame([{
    "nct_id": nct_id,
    "review_type": "PHASE47_BOTANICAL_SCOPE_CHECK",
    "initial_category": "POST_MECHANISM_SCOPE_AUDIT",
    "final_decision": "EXCLUDE",
    "decision_confidence": "high",
    "decision_reason": reason,
    "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
}])

review_decisions = pd.concat(
    [review_decisions, new_review],
    ignore_index=True
)

if review_decisions["nct_id"].duplicated().any():
    raise ValueError("Duplicate NCT IDs after SKF7 review-file patch.")

review_decisions.to_csv(review_decisions_file, index=False)

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

if len(included_ids) != 1000:
    raise ValueError(
        f"Expected 1000 included trials after SKF7 exclusion; "
        f"found {len(included_ids)}."
    )

trials = trials[trials["nct_id"].isin(included_ids)].copy()
interventions = interventions[
    interventions["nct_id"].isin(included_ids)
].copy()
locations = locations[locations["nct_id"].isin(included_ids)].copy()
collaborators = collaborators[
    collaborators["nct_id"].isin(included_ids)
].copy()

trials.to_csv(trials_file, index=False)
interventions.to_csv(interventions_file, index=False)
locations.to_csv(locations_file, index=False)
collaborators.to_csv(collaborators_file, index=False)

roles.loc[
    roles["nct_id"] == nct_id,
    "final_role"
] = "OUT_OF_SCOPE"

if "role_review_confidence_final" in roles.columns:
    roles.loc[
        roles["nct_id"] == nct_id,
        "role_review_confidence_final"
    ] = "high"

if "role_review_reason_final" in roles.columns:
    roles.loc[
        roles["nct_id"] == nct_id,
        "role_review_reason_final"
    ] = reason

roles.to_csv(roles_file, index=False)

analysis_ready = roles[
    roles["nct_id"].isin(included_ids)
    & ~roles["final_role"].isin(
        ["CONTROL_PLACEBO", "OUT_OF_SCOPE"]
    )
].copy()

analysis_ready.to_csv(analysis_ready_file, index=False)

focal = analysis_ready[
    analysis_ready["final_role"] == "FOCAL_OBESITY_THERAPY"
].copy()

focal_map = (
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

focal_map.to_csv(focal_map_file, index=False)

missing = included_ids - set(focal_map["nct_id"])

skf = focal_map[
    focal_map["canonical_name_final"]
    .fillna("")
    .astype(str)
    .str.contains("SKF7", case=False, regex=False)
]

print()
print("Final trial decisions:")
print(trial_decisions["final_decision"].value_counts())

print()
print("=== Focal-map validation ===")
print("Included trials:", len(included_ids))
print("Unique trial × focal-therapy pairs:", len(focal_map))
print("Trials represented:", focal_map["nct_id"].nunique())
print("Included trials with no focal therapy:", len(missing))
print("SKF7 focal pairs remaining:", len(skf))

if missing:
    raise ValueError("At least one included trial has no focal therapy.")

if len(skf):
    raise ValueError("SKF7 remains in focal map.")

print()
print("SKF7 botanical scope correction complete.")
