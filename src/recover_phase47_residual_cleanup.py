from pathlib import Path
import pandas as pd

processed = Path("data/processed")

roles_file = processed / "obesity_intervention_roles_final.csv"
analysis_ready_file = processed / "obesity_interventions_analysis_ready.csv"
focal_map_file = processed / "obesity_trial_focal_therapy_map.csv"

trial_decisions_file = processed / "trial_decisions_final.csv"

trials_file = processed / "obesity_trials_clean.csv"
interventions_file = processed / "obesity_interventions_clean.csv"
locations_file = processed / "obesity_locations_clean.csv"
collaborators_file = processed / "obesity_collaborators_clean.csv"

roles = pd.read_csv(roles_file)
trial_decisions = pd.read_csv(trial_decisions_file)

trials = pd.read_csv(trials_file)
interventions = pd.read_csv(interventions_file)
locations = pd.read_csv(locations_file)
collaborators = pd.read_csv(collaborators_file)

print("=== Phase 4.7 residual cleanup recovery ===")

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
        for p in patterns:
            mask = mask | s.str.contains(p.lower(), regex=False)
    return mask

# Re-apply intended role corrections. Previous script stopped before roles were saved.
coc_mask = (
    (roles["nct_id"] == "NCT07523711")
    & contains_any(roles, ["coc", "combined oral contraceptive"])
)

glucose_mask = (
    (roles["nct_id"] == "NCT05713773")
    & contains_any(
        roles,
        ["coated beads glucose", "uncoated beads glucose"]
    )
)

dd01_mask = (
    (roles["nct_id"] == "NCT04812262")
    & contains_any(roles, ["dd01"])
)

print("COC rows corrected:", int(coc_mask.sum()))
print("Glucose-bead rows corrected:", int(glucose_mask.sum()))
print("DD01 rows corrected:", int(dd01_mask.sum()))

if coc_mask.sum() == 0:
    raise ValueError("Could not find the COC row in NCT07523711.")
if glucose_mask.sum() == 0:
    raise ValueError("Could not find glucose-bead rows in NCT05713773.")
if dd01_mask.sum() == 0:
    raise ValueError("Could not find DD01 rows in NCT04812262.")

roles.loc[coc_mask, "final_role"] = "BACKGROUND_OR_CONCOMITANT"
roles.loc[glucose_mask, "final_role"] = "OUT_OF_SCOPE"
roles.loc[dd01_mask, "final_role"] = "OUT_OF_SCOPE"

if "role_review_confidence_final" in roles.columns:
    roles.loc[
        coc_mask | glucose_mask | dd01_mask,
        "role_review_confidence_final"
    ] = "high"

# Only two trials should be excluded. COC trial remains included.
exclude_ids = {"NCT04812262", "NCT05713773"}

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"] == "NCT07523711",
    "final_decision"
] = "INCLUDE"

trial_decisions.to_csv(trial_decisions_file, index=False)

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

print("Included trials after recovery:", len(included_ids))

if len(included_ids) != 991:
    raise ValueError(
        f"Expected 991 included trials after recovery; found {len(included_ids)}."
    )

# Validate retained COC trial still has another focal therapy.
coc_trial_focal = roles[
    (roles["nct_id"] == "NCT07523711")
    & (roles["final_role"] == "FOCAL_OBESITY_THERAPY")
].copy()

print()
print("Remaining focal therapies in NCT07523711:")
display_cols = [
    c for c in [
        "canonical_name_final",
        "analysis_name_candidate",
        "final_role",
    ]
    if c in coc_trial_focal.columns
]
print(coc_trial_focal[display_cols].to_string(index=False))

if coc_trial_focal.empty:
    raise ValueError(
        "NCT07523711 has no focal obesity therapy after COC correction."
    )

# Save corrected roles.
roles.loc[
    roles["nct_id"].isin(exclude_ids),
    "final_role"
] = "OUT_OF_SCOPE"
roles.to_csv(roles_file, index=False)

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

# Rebuild analysis-ready interventions and focal map.
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

missing_focal = included_ids - set(focal_map["nct_id"])

print()
print("=== Recovered Phase 4.7 validation ===")
print("Included trials:", len(included_ids))
print("Analysis-ready intervention rows:", len(analysis_ready))
print("Focal therapy rows:", len(focal))
print("Unique trial × focal-therapy pairs:", len(focal_map))
print("Trials represented:", focal_map["nct_id"].nunique())
print("Included trials with no focal therapy:", len(missing_focal))

for label, term in {
    "COC": "coc",
    "coated glucose": "coated beads glucose",
    "DD01": "dd01",
}.items():
    matches = focal_map[
        focal_map["canonical_name_final"]
        .fillna("")
        .astype(str)
        .str.contains(term, case=False, regex=False)
    ]
    print(f"{label} focal pairs remaining:", len(matches))

if missing_focal:
    print(sorted(missing_focal))
    raise ValueError("At least one included trial lacks a focal therapy.")

for nct_id in exclude_ids:
    if nct_id in set(focal_map["nct_id"]):
        raise ValueError(f"{nct_id} still appears in focal map.")

print()
print(
    "RECOVERY COMPLETE: 991 included trials and all included "
    "trials retain at least one focal obesity therapy."
)
