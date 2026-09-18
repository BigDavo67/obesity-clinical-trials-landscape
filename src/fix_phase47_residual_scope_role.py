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

print("=== Phase 4.7 residual scope/role cleanup ===")

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

def exact_or_contains_mask(df, patterns):
    mask = pd.Series(False, index=df.index)
    for col in name_cols:
        s = df[col].fillna("").astype(str).str.strip().str.lower()
        for p in patterns:
            p = p.lower()
            mask = mask | s.eq(p) | s.str.contains(p, regex=False)
    return mask

# A) Combined oral contraceptive (COC)
coc_mask = (
    (roles["final_role"] == "FOCAL_OBESITY_THERAPY")
    & exact_or_contains_mask(
        roles,
        ["coc", "combined oral contraceptive"]
    )
)

# B) Aphaia coated glucose-bead nutrient formulations
glucose_bead_mask = (
    (roles["final_role"] == "FOCAL_OBESITY_THERAPY")
    & exact_or_contains_mask(
        roles,
        [
            "coated beads glucose",
            "uncoated beads glucose",
        ]
    )
)

# C) DD01 — MASLD/MASH programme, obesity is population rather than target
dd01_mask = (
    (roles["final_role"] == "FOCAL_OBESITY_THERAPY")
    & exact_or_contains_mask(roles, ["dd01"])
)

affected = {
    "COC": set(roles.loc[coc_mask, "nct_id"]),
    "COATED_GLUCOSE_BEADS": set(
        roles.loc[glucose_bead_mask, "nct_id"]
    ),
    "DD01": set(roles.loc[dd01_mask, "nct_id"]),
}

for key, ids in affected.items():
    print(f"{key} focal rows:", len(ids), "trial(s)", sorted(ids))

# Role corrections
roles.loc[coc_mask, "final_role"] = "BACKGROUND_OR_CONCOMITANT"
roles.loc[glucose_bead_mask, "final_role"] = "OUT_OF_SCOPE"
roles.loc[dd01_mask, "final_role"] = "OUT_OF_SCOPE"

if "role_review_confidence_final" in roles.columns:
    roles.loc[
        coc_mask | glucose_bead_mask | dd01_mask,
        "role_review_confidence_final"
    ] = "high"

# After role corrections, exclude only affected trials that have no
# genuine focal obesity therapy remaining.
all_affected_ids = set().union(*affected.values())

remaining_focal_ids = set(
    roles.loc[
        roles["final_role"] == "FOCAL_OBESITY_THERAPY",
        "nct_id"
    ]
)

exclude_ids = {
    nct_id for nct_id in all_affected_ids
    if nct_id not in remaining_focal_ids
}

print()
print("Trials losing all focal obesity therapies:", sorted(exclude_ids))

reason_map = {}

for nct_id in exclude_ids:
    if nct_id in affected["COC"]:
        reason_map[nct_id] = (
            "Combined oral contraceptive study evaluates effects of "
            "contraception on weight/body composition rather than treating obesity."
        )
    elif nct_id in affected["COATED_GLUCOSE_BEADS"]:
        reason_map[nct_id] = (
            "Coated glucose-bead intervention is a nutrient-formulation "
            "PK/PD programme and falls outside the frozen conventional "
            "pharmacotherapy scope."
        )
    elif nct_id in affected["DD01"]:
        reason_map[nct_id] = (
            "DD01 is a MASLD/MASH development programme conducted in "
            "overweight/obese participants rather than an obesity-therapy programme."
        )

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_source"
] = "PHASE47_RESIDUAL_SCOPE_ROLE_CLEANUP"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_reason"
] = (
    trial_decisions.loc[
        trial_decisions["nct_id"].isin(exclude_ids),
        "nct_id"
    ].map(reason_map)
)

trial_decisions.to_csv(trial_decisions_file, index=False)

# Patch review log.
review_decisions = review_decisions[
    ~review_decisions["nct_id"].isin(exclude_ids)
].copy()

if exclude_ids:
    new_rows = pd.DataFrame({
        "nct_id": list(exclude_ids),
        "review_type": "PHASE47_RESIDUAL_SCOPE_ROLE_CLEANUP",
        "initial_category": "POST_MECHANISM_SCOPE_AUDIT",
        "final_decision": "EXCLUDE",
        "decision_confidence": "high",
        "decision_reason": [
            reason_map[nct_id] for nct_id in exclude_ids
        ],
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
        "Duplicate NCT IDs after residual cleanup."
    )

review_decisions.to_csv(review_decisions_file, index=False)

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

# We expect 993 before this script and three residual false-positive
# trials, giving 990, but report dynamically as well.
print("Included trials after cleanup:", len(included_ids))

if len(included_ids) != 990:
    raise ValueError(
        f"Expected 990 included trials after cleanup; "
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

# All rows from excluded trials become OUT_OF_SCOPE.
roles.loc[
    roles["nct_id"].isin(exclude_ids),
    "final_role"
] = "OUT_OF_SCOPE"

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
    .sort_values(["nct_id", "canonical_name_final"])
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

for term in [
    "coc",
    "coated beads glucose",
    "dd01",
]:
    matches = focal_map[
        focal_map["canonical_name_final"]
        .fillna("")
        .astype(str)
        .str.contains(term, case=False, regex=False)
    ]
    print(f"{term} focal pairs remaining:", len(matches))

if missing_focal:
    raise ValueError(
        "At least one included trial lacks a focal obesity therapy."
    )

print()
print("Residual Phase 4.7 scope/role cleanup complete.")
