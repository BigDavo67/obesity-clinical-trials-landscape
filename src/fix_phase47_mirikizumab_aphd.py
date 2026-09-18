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

print("=== Phase 4.7 role/scope correction ===")


# ------------------------------------------------------------
# 1. MIRIKIZUMAB ROLE CORRECTION
# ------------------------------------------------------------

mirikizumab_trials = {
    "NCT06937086": (
        "Mirikizumab treats ulcerative colitis; tirzepatide is the "
        "weight-loss/obesity therapy in COMMIT-UC."
    ),
    "NCT06937099": (
        "Mirikizumab treats Crohn's disease; tirzepatide is the "
        "weight-loss/obesity therapy in COMMIT-CD."
    ),
}

name_cols = [
    col for col in [
        "canonical_name_final",
        "analysis_name_candidate",
        "canonical_candidate",
        "active_name_key",
        "active_component_raw",
        "active_component_source",
        "intervention_name",
    ]
    if col in roles.columns
]

def contains_therapy(df, therapy):
    mask = pd.Series(False, index=df.index)
    for col in name_cols:
        mask = mask | (
            df[col]
            .fillna("")
            .astype(str)
            .str.contains(therapy, case=False, regex=False)
        )
    return mask

miri_ids = set(mirikizumab_trials)

miri_mask = (
    roles["nct_id"].isin(miri_ids)
    & contains_therapy(roles, "mirikizumab")
)

miri_rows = roles.loc[miri_mask].copy()

print("Mirikizumab rows found:", len(miri_rows))

if miri_rows["nct_id"].nunique() != 2:
    raise ValueError(
        "Expected mirikizumab rows in exactly two COMMIT trials."
    )

roles.loc[
    miri_mask,
    "final_role"
] = "BACKGROUND_OR_CONCOMITANT"

if "role_review_confidence_final" in roles.columns:
    roles.loc[
        miri_mask,
        "role_review_confidence_final"
    ] = "high"

if "role_review_reason_final" in roles.columns:
    roles.loc[
        miri_mask,
        "role_review_reason_final"
    ] = (
        roles.loc[miri_mask, "nct_id"]
        .map(mirikizumab_trials)
    )

# Validate focal tirzepatide remains.
for nct_id in sorted(miri_ids):
    subset = roles[roles["nct_id"] == nct_id]

    tirz_mask = contains_therapy(subset, "tirzepatide")

    focal_tirz = subset[
        tirz_mask
        & (subset["final_role"] == "FOCAL_OBESITY_THERAPY")
    ]

    if focal_tirz.empty:
        raise ValueError(
            f"{nct_id} does not retain focal tirzepatide."
        )


# ------------------------------------------------------------
# 2. APHD SCOPE EXCLUSIONS
# ------------------------------------------------------------

aphd_exclusions = {
    "NCT05385978": (
        "APHD-012 is distal jejunal-release dextrose beads. "
        "Although registered as a drug and tested for obesity, it is "
        "a nutrient/glucose formulation designed to trigger endogenous "
        "gut nutrient sensing, which falls outside the frozen conventional "
        "pharmacotherapy scope excluding food-derived/nutrient interventions."
    ),
    "NCT07008456": (
        "APHD-012/APHD-002 are distal jejunal-release dextrose-bead "
        "formulations. They are nutrient/glucose formulations rather than "
        "conventional pharmacological obesity assets and therefore fall "
        "outside the frozen project scope."
    ),
}

exclude_ids = set(aphd_exclusions)

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_source"
] = "PHASE47_NUTRIENT_FORMULATION_SCOPE_CHECK"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_reason"
] = (
    trial_decisions.loc[
        trial_decisions["nct_id"].isin(exclude_ids),
        "nct_id"
    ]
    .map(aphd_exclusions)
)

trial_decisions.to_csv(
    trial_decisions_file,
    index=False
)

# Patch review_decisions_final for reproducibility.
review_decisions = review_decisions[
    ~review_decisions["nct_id"].isin(exclude_ids)
].copy()

scope_rows = pd.DataFrame({
    "nct_id": list(exclude_ids),
    "review_type": "PHASE47_NUTRIENT_FORMULATION_SCOPE_CHECK",
    "initial_category": "POST_MECHANISM_SCOPE_AUDIT",
    "final_decision": "EXCLUDE",
    "decision_confidence": "high",
    "decision_reason": [
        aphd_exclusions[nct]
        for nct in exclude_ids
    ],
    "source_url": [
        f"https://clinicaltrials.gov/study/{nct}"
        for nct in exclude_ids
    ],
})

review_decisions = pd.concat(
    [review_decisions, scope_rows],
    ignore_index=True
)

if review_decisions["nct_id"].duplicated().any():
    raise ValueError(
        "Duplicate NCT IDs after patching review_decisions_final.csv."
    )

review_decisions.to_csv(
    review_decisions_file,
    index=False
)


# ------------------------------------------------------------
# 3. FILTER CLEAN RELATIONAL TABLES
# ------------------------------------------------------------

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

if len(included_ids) != 1001:
    raise ValueError(
        f"Expected 1001 included trials after APHD exclusions, "
        f"found {len(included_ids)}."
    )

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


# ------------------------------------------------------------
# 4. MARK EXCLUDED TRIAL ROLE ROWS OUT OF SCOPE
# ------------------------------------------------------------

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
        ]
        .map(aphd_exclusions)
    )

roles.to_csv(
    roles_file,
    index=False
)


# ------------------------------------------------------------
# 5. REBUILD ANALYSIS-READY + FOCAL MAP
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 6. FINAL VALIDATION
# ------------------------------------------------------------

missing_focal = (
    included_ids
    - set(focal_map["nct_id"])
)

mirikizumab_focal = focal_map[
    focal_map["canonical_name_final"]
    .fillna("")
    .astype(str)
    .str.contains(
        "mirikizumab",
        case=False,
        regex=False
    )
]

aphd_focal = focal_map[
    focal_map["canonical_name_final"]
    .fillna("")
    .astype(str)
    .str.contains(
        "aphd",
        case=False,
        regex=False
    )
]

print()
print("Final trial decisions:")
print(
    trial_decisions["final_decision"]
    .value_counts()
)

print()
print("=== Corrected Phase 4.7 focal-map validation ===")
print("Included trials:", len(included_ids))
print("Focal therapy rows:", len(focal))
print("Unique trial × focal-therapy pairs:", len(focal_map))
print("Trials represented:", focal_map["nct_id"].nunique())
print("Included trials with no focal therapy:", len(missing_focal))
print("Mirikizumab focal pairs remaining:", len(mirikizumab_focal))
print("APHD focal pairs remaining:", len(aphd_focal))

if missing_focal:
    print(sorted(missing_focal))
    raise ValueError(
        "At least one included trial lacks a focal therapy."
    )

if len(mirikizumab_focal) != 0:
    raise ValueError(
        "Mirikizumab still appears as a focal obesity therapy."
    )

if len(aphd_focal) != 0:
    raise ValueError(
        "APHD nutrient formulation still appears in focal map."
    )

print()
print("Phase 4.7 role/scope correction complete.")
