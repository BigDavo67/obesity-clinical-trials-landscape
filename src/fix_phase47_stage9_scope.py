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

print("=== Phase 4.7 Stage-9 residual scope cleanup ===")

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

def contains_any(df, patterns):
    mask = pd.Series(False, index=df.index)
    for col in name_cols:
        s = df[col].fillna("").astype(str).str.strip().str.lower()
        for pattern in patterns:
            mask = mask | s.str.contains(pattern.lower(), regex=False)
    return mask

scope_groups = {
    "FMT_NONPHARMA": {
        "patterns": ["fecal microbiota transplantation"],
        "reason": (
            "Fecal microbiota transplantation is a non-pharmacological "
            "microbiome procedure/biological intervention and falls outside "
            "the frozen conventional pharmacotherapy scope."
        ),
    },
    "MULBERRY_TRADITIONAL": {
        "patterns": ["mulberry twig alkaloids"],
        "reason": (
            "Mulberry twig alkaloids are a traditional/plant-derived "
            "intervention primarily developed for glycaemic control rather "
            "than a conventional obesity pharmacotherapy programme."
        ),
    },
    "GLUCOSE_BEADS_NUTRIENT": {
        "patterns": [
            "glucose coated beads",
            "glucose uncoated beads",
        ],
        "reason": (
            "Coated/uncoated glucose beads are nutrient-formulation "
            "interventions and fall outside the frozen conventional "
            "pharmacotherapy scope."
        ),
    },
    "GLY_LOW_SUPPLEMENT": {
        "patterns": ["gly-low"],
        "reason": (
            "GLY-Low is a supplementation mixture of GRAS compounds "
            "(including vitamins/nutrient-derived compounds) rather than a "
            "conventional obesity pharmacotherapy asset."
        ),
    },
    "FT4101_COMORBIDITY": {
        "patterns": ["ft-4101"],
        "reason": (
            "FT-4101 is a fatty-acid-synthase inhibitor developed for "
            "NAFLD/NASH/steatosis; overweight/obesity is the study population "
            "rather than obesity being the therapeutic indication."
        ),
    },
}

all_exclude_ids = set()
reason_map = {}

for label, cfg in scope_groups.items():
    mask = (
        (roles["final_role"] == "FOCAL_OBESITY_THERAPY")
        & contains_any(roles, cfg["patterns"])
    )

    ids = sorted(set(roles.loc[mask, "nct_id"]))

    print(f"{label}: {len(ids)} trial(s) -> {ids}")

    for nct_id in ids:
        all_exclude_ids.add(nct_id)
        reason_map[nct_id] = cfg["reason"]

if not all_exclude_ids:
    print("No matching residual scope rows found.")
    raise SystemExit(0)

print()
print("Trials to exclude:", sorted(all_exclude_ids))

trial_decisions.loc[
    trial_decisions["nct_id"].isin(all_exclude_ids),
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(all_exclude_ids),
    "final_decision_source"
] = "PHASE47_STAGE9_SCOPE_CLEANUP"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(all_exclude_ids),
    "final_decision_reason"
] = (
    trial_decisions.loc[
        trial_decisions["nct_id"].isin(all_exclude_ids),
        "nct_id"
    ].map(reason_map)
)

trial_decisions.to_csv(trial_decisions_file, index=False)

# Patch review log.
review_decisions = review_decisions[
    ~review_decisions["nct_id"].isin(all_exclude_ids)
].copy()

new_rows = pd.DataFrame({
    "nct_id": list(all_exclude_ids),
    "review_type": "PHASE47_STAGE9_SCOPE_CLEANUP",
    "initial_category": "POST_MECHANISM_SCOPE_AUDIT",
    "final_decision": "EXCLUDE",
    "decision_confidence": "high",
    "decision_reason": [
        reason_map[nct_id]
        for nct_id in all_exclude_ids
    ],
    "source_url": [
        f"https://clinicaltrials.gov/study/{nct_id}"
        for nct_id in all_exclude_ids
    ],
})

review_decisions = pd.concat(
    [review_decisions, new_rows],
    ignore_index=True
)

if review_decisions["nct_id"].duplicated().any():
    raise ValueError(
        "Duplicate NCT IDs after Stage-9 scope cleanup."
    )

review_decisions.to_csv(review_decisions_file, index=False)

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

# Mark all role rows from excluded trials out of scope.
roles.loc[
    roles["nct_id"].isin(all_exclude_ids),
    "final_role"
] = "OUT_OF_SCOPE"

if "role_review_confidence_final" in roles.columns:
    roles.loc[
        roles["nct_id"].isin(all_exclude_ids),
        "role_review_confidence_final"
    ] = "high"

if "role_review_reason_final" in roles.columns:
    roles.loc[
        roles["nct_id"].isin(all_exclude_ids),
        "role_review_reason_final"
    ] = (
        roles.loc[
            roles["nct_id"].isin(all_exclude_ids),
            "nct_id"
        ].map(reason_map)
    )

roles.to_csv(roles_file, index=False)

# Filter relational tables.
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

# Rebuild analysis-ready table and focal map.
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
    .sort_values(["nct_id", "canonical_name_final"])
)

focal_map.to_csv(focal_map_file, index=False)

missing_focal = included_ids - set(focal_map["nct_id"])

print()
print("=== Stage-9 scope cleanup validation ===")
print("Included trials:", len(included_ids))
print("Analysis-ready intervention rows:", len(analysis_ready))
print("Focal therapy rows:", len(focal))
print("Unique trial × focal-therapy pairs:", len(focal_map))
print("Trials represented:", focal_map["nct_id"].nunique())
print("Included trials with no focal therapy:", len(missing_focal))

for label, cfg in scope_groups.items():
    mask = pd.Series(False, index=focal_map.index)
    s = (
        focal_map["canonical_name_final"]
        .fillna("")
        .astype(str)
        .str.lower()
    )
    for pattern in cfg["patterns"]:
        mask = mask | s.str.contains(pattern.lower(), regex=False)
    print(f"{label} focal pairs remaining:", int(mask.sum()))

if missing_focal:
    print(sorted(missing_focal))
    raise ValueError(
        "At least one included trial lacks a focal obesity therapy."
    )

print()
print("Stage-9 residual scope cleanup complete.")
