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

roles = pd.read_csv(roles_file)
trial_decisions = pd.read_csv(trial_decisions_file)
review_decisions = pd.read_csv(review_decisions_file)

trials = pd.read_csv(trials_file)
interventions = pd.read_csv(interventions_file)
locations = pd.read_csv(locations_file)
collaborators = pd.read_csv(collaborators_file)

print("=== Phase 4.7 drospirenone residual scope/role cleanup ===")

name_cols = [
    c for c in [
        "canonical_name_final",
        "analysis_name_candidate",
        "canonical_candidate",
        "active_name_key",
        "active_component_raw",
        "active_component_source",
        "intervention_name",
    ]
    if c in roles.columns
]

drosp_mask = pd.Series(False, index=roles.index)

for col in name_cols:
    s = roles[col].fillna("").astype(str).str.strip().str.lower()
    drosp_mask = drosp_mask | s.eq("drospirenone")

drosp_mask = drosp_mask & (
    roles["final_role"] == "FOCAL_OBESITY_THERAPY"
)

drosp_rows = roles.loc[drosp_mask].copy()

print("Drospirenone focal rows found:", len(drosp_rows))

if drosp_rows.empty:
    print("No focal drospirenone rows remain; nothing to change.")
    raise SystemExit(0)

affected_ids = set(drosp_rows["nct_id"])

print("Affected trials:", sorted(affected_ids))

# Drospirenone is contraception therapy in these obesity-population studies,
# not an obesity-development therapy.
roles.loc[
    drosp_mask,
    "final_role"
] = "BACKGROUND_OR_CONCOMITANT"

if "role_review_confidence_final" in roles.columns:
    roles.loc[
        drosp_mask,
        "role_review_confidence_final"
    ] = "high"

if "role_review_reason_final" in roles.columns:
    roles.loc[
        drosp_mask,
        "role_review_reason_final"
    ] = (
        "Drospirenone is being evaluated for contraception/pharmacokinetics "
        "in participants with obesity, not as a treatment for obesity."
    )

# Exclude a trial only if it has no genuine focal obesity therapy left.
remaining_focal_ids = set(
    roles.loc[
        roles["final_role"] == "FOCAL_OBESITY_THERAPY",
        "nct_id"
    ]
)

exclude_ids = {
    nct_id
    for nct_id in affected_ids
    if nct_id not in remaining_focal_ids
}

print("Trials losing all focal obesity therapies:", sorted(exclude_ids))

reason = (
    "Drospirenone is a contraception intervention studied in participants "
    "with obesity, rather than an obesity-treatment or obesity-development asset."
)

if exclude_ids:
    trial_decisions.loc[
        trial_decisions["nct_id"].isin(exclude_ids),
        "final_decision"
    ] = "EXCLUDE"

    trial_decisions.loc[
        trial_decisions["nct_id"].isin(exclude_ids),
        "final_decision_source"
    ] = "PHASE47_DROSPIRENONE_SCOPE_CHECK"

    trial_decisions.loc[
        trial_decisions["nct_id"].isin(exclude_ids),
        "final_decision_reason"
    ] = reason

trial_decisions.to_csv(
    trial_decisions_file,
    index=False
)

# Update review log.
if exclude_ids:
    review_decisions = review_decisions[
        ~review_decisions["nct_id"].isin(exclude_ids)
    ].copy()

    new_rows = pd.DataFrame({
        "nct_id": list(exclude_ids),
        "review_type": "PHASE47_DROSPIRENONE_SCOPE_CHECK",
        "initial_category": "POST_MECHANISM_SCOPE_AUDIT",
        "final_decision": "EXCLUDE",
        "decision_confidence": "high",
        "decision_reason": reason,
        "source_url": [
            f"https://clinicaltrials.gov/study/{nct_id}"
            for nct_id in exclude_ids
        ],
    })

    review_decisions = pd.concat(
        [review_decisions, new_rows],
        ignore_index=True
    )

    if review_decisions["nct_id"].duplicated().any():
        raise ValueError(
            "Duplicate NCT IDs after drospirenone scope patch."
        )

    review_decisions.to_csv(
        review_decisions_file,
        index=False
    )

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

# Mark all rows from newly excluded trials out of scope.
if exclude_ids:
    roles.loc[
        roles["nct_id"].isin(exclude_ids),
        "final_role"
    ] = "OUT_OF_SCOPE"

roles.to_csv(
    roles_file,
    index=False
)

# Filter clean relational tables.
trials = trials[
    trials["nct_id"].isin(included_ids)
].copy()

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

# Rebuild analysis-ready table.
analysis_ready = roles[
    roles["nct_id"].isin(included_ids)
    & ~roles["final_role"].isin(
        ["CONTROL_PLACEBO", "OUT_OF_SCOPE"]
    )
].copy()

analysis_ready.to_csv(
    analysis_ready_file,
    index=False
)

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

focal_map.to_csv(
    focal_map_file,
    index=False
)

missing_focal = included_ids - set(focal_map["nct_id"])

drosp_focal = focal_map[
    focal_map["canonical_name_final"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("drospirenone")
]

print()
print("=== Drospirenone cleanup validation ===")
print("Included trials:", len(included_ids))
print("Unique trial × focal-therapy pairs:", len(focal_map))
print("Trials represented:", focal_map["nct_id"].nunique())
print("Included trials with no focal therapy:", len(missing_focal))
print("Drospirenone focal pairs remaining:", len(drosp_focal))

if missing_focal:
    raise ValueError(
        "An included trial lacks a focal obesity therapy."
    )

if len(drosp_focal) != 0:
    raise ValueError(
        "Drospirenone still appears as a focal obesity therapy."
    )

print()
print("Drospirenone cleanup complete.")
