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

exclusions = {
    "NCT05463783": (
        "Biktarvy and Symtuza are antiretroviral HIV regimens. "
        "The study investigates mechanisms of ARV-associated weight gain; "
        "neither intervention is being developed as an obesity therapy."
    ),
    "NCT05966727": (
        "Cholestyramine is being used to lower persistent organic pollutant "
        "levels in obese women before bariatric surgery. The therapeutic "
        "objective is pollutant elimination, not obesity treatment."
    ),
    "NCT04874701": (
        "Capsimax is a proprietary capsicum/capsaicinoid dietary supplement. "
        "It falls outside the frozen conventional pharmacotherapy scope "
        "excluding nutraceutical and food-derived interventions."
    ),
}

roles = pd.read_csv(roles_file)
trial_decisions = pd.read_csv(trial_decisions_file)
review_decisions = pd.read_csv(review_decisions_file)

trials = pd.read_csv(trials_file)
interventions = pd.read_csv(interventions_file)
locations = pd.read_csv(locations_file)
collaborators = pd.read_csv(collaborators_file)

exclude_ids = set(exclusions)

print("=== Phase 4.7 final long-tail scope correction ===")
print("Trials to exclude:", len(exclude_ids))

missing = exclude_ids - set(trial_decisions["nct_id"])
if missing:
    raise ValueError(
        f"Scope-correction IDs not found: {sorted(missing)}"
    )

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_source"
] = "PHASE47_FINAL_LONGTAIL_SCOPE_CHECK"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_reason"
] = (
    trial_decisions.loc[
        trial_decisions["nct_id"].isin(exclude_ids),
        "nct_id"
    ].map(exclusions)
)

trial_decisions.to_csv(trial_decisions_file, index=False)

# Patch reproducibility file.
review_decisions = review_decisions[
    ~review_decisions["nct_id"].isin(exclude_ids)
].copy()

new_review_rows = pd.DataFrame({
    "nct_id": list(exclude_ids),
    "review_type": "PHASE47_FINAL_LONGTAIL_SCOPE_CHECK",
    "initial_category": "POST_MECHANISM_SCOPE_AUDIT",
    "final_decision": "EXCLUDE",
    "decision_confidence": "high",
    "decision_reason": [exclusions[nct] for nct in exclude_ids],
    "source_url": [
        f"https://clinicaltrials.gov/study/{nct}"
        for nct in exclude_ids
    ],
})

review_decisions = pd.concat(
    [review_decisions, new_review_rows],
    ignore_index=True
)

if review_decisions["nct_id"].duplicated().any():
    raise ValueError(
        "Duplicate NCT IDs after final long-tail scope patch."
    )

review_decisions.to_csv(review_decisions_file, index=False)

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

if len(included_ids) != 993:
    raise ValueError(
        f"Expected 993 included trials after three exclusions; "
        f"found {len(included_ids)}."
    )

# Filter clean relational tables.
trials = trials[trials["nct_id"].isin(included_ids)].copy()
interventions = interventions[
    interventions["nct_id"].isin(included_ids)
].copy()
locations = locations[
    locations["nct_id"].isin(included_ids)
].copy()
collaborators = collaborators[
    collaborators["nct_id"].isin(included_ids)
].copy()

trials.to_csv(trials_file, index=False)
interventions.to_csv(interventions_file, index=False)
locations.to_csv(locations_file, index=False)
collaborators.to_csv(collaborators_file, index=False)

# Mark role rows out of scope.
roles.loc[
    roles["nct_id"].isin(exclude_ids),
    "final_role"
] = "OUT_OF_SCOPE"

if "role_review_confidence_final" in roles.columns:
    roles.loc[
        roles["nct_id"].isin(exclude_ids),
        "role_review_confidence_final"
    ] = "high"

if "role_review_reason_final" in roles.columns:
    roles.loc[
        roles["nct_id"].isin(exclude_ids),
        "role_review_reason_final"
    ] = (
        roles.loc[
            roles["nct_id"].isin(exclude_ids),
            "nct_id"
        ].map(exclusions)
    )

roles.to_csv(roles_file, index=False)

# Rebuild analysis-ready table.
analysis_ready = roles[
    roles["nct_id"].isin(included_ids)
    & ~roles["final_role"].isin(
        ["CONTROL_PLACEBO", "OUT_OF_SCOPE"]
    )
].copy()

analysis_ready.to_csv(analysis_ready_file, index=False)

# Rebuild focal map.
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
    .sort_values(
        ["nct_id", "canonical_name_final"]
    )
)

focal_map.to_csv(focal_map_file, index=False)

missing_focal = included_ids - set(focal_map["nct_id"])

print()
print("Final trial decisions:")
print(trial_decisions["final_decision"].value_counts())

print()
print("=== Revised focal-map validation ===")
print("Included trials:", len(included_ids))
print("Unique trial × focal-therapy pairs:", len(focal_map))
print("Trials represented:", focal_map["nct_id"].nunique())
print("Included trials with no focal therapy:", len(missing_focal))

for term in ["biktarvy", "cholestyramine", "capsimax"]:
    matches = focal_map[
        focal_map["canonical_name_final"]
        .fillna("")
        .astype(str)
        .str.contains(term, case=False, regex=False)
    ]
    print(f"{term} focal pairs remaining:", len(matches))

if missing_focal:
    print(sorted(missing_focal))
    raise ValueError("An included trial has no focal therapy.")

print()
print("Final long-tail scope correction complete.")
