from pathlib import Path
import pandas as pd

processed = Path("data/processed")

resolution_file = processed / "phase46_missing_focal_resolution.csv"

trial_decisions_file = processed / "trial_decisions_final.csv"
review_decisions_file = processed / "review_decisions_final.csv"

trials_file = processed / "obesity_trials_clean.csv"
interventions_file = processed / "obesity_interventions_clean.csv"
locations_file = processed / "obesity_locations_clean.csv"
collaborators_file = processed / "obesity_collaborators_clean.csv"

roles_file = processed / "obesity_intervention_roles_final.csv"

analysis_ready_file = processed / "obesity_interventions_analysis_ready.csv"
focal_map_file = processed / "obesity_trial_focal_therapy_map.csv"


# ============================================================
# 1. LOAD
# ============================================================

resolution = pd.read_csv(resolution_file)
trial_decisions = pd.read_csv(trial_decisions_file)
review_decisions = pd.read_csv(review_decisions_file)

trials = pd.read_csv(trials_file)
interventions = pd.read_csv(interventions_file)
locations = pd.read_csv(locations_file)
collaborators = pd.read_csv(collaborators_file)

roles = pd.read_csv(roles_file)

print("=== Phase 4.6 missing-focal resolution ===")
print("Resolution rows:", len(resolution))


# ============================================================
# 2. RESOLVE DECISION SETS
# ============================================================

exclude_df = resolution[
    resolution["action"] == "EXCLUDE_FROM_SCOPE"
].copy()

focal_df = resolution[
    resolution["action"] == "KEEP_AND_ASSIGN_FOCAL"
].copy()

exclude_ids = set(exclude_df["nct_id"])
focal_map = focal_df.set_index("nct_id")["focal_therapy"].to_dict()
reason_map = resolution.set_index("nct_id")["reason"].to_dict()
source_map = resolution.set_index("nct_id")["source_url"].to_dict()

print("Late exclusions:", len(exclude_ids))
print("Missing focal assignments:", len(focal_map))


# ============================================================
# 3. PATCH TRIAL DECISIONS FOR THE THREE LATE EXCLUSIONS
# ============================================================

unknown_exclude_ids = exclude_ids - set(trial_decisions["nct_id"])

if unknown_exclude_ids:
    raise ValueError(
        f"Scope-correction IDs absent from trial_decisions_final.csv: "
        f"{sorted(unknown_exclude_ids)}"
    )

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision"
] = "EXCLUDE"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_source"
] = "PHASE46_MISSING_FOCAL_SCOPE_CHECK"

trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "final_decision_reason"
] = trial_decisions.loc[
    trial_decisions["nct_id"].isin(exclude_ids),
    "nct_id"
].map(reason_map)

trial_decisions.to_csv(
    trial_decisions_file,
    index=False
)


# ============================================================
# 4. PATCH REVIEW_DECISIONS_FINAL FOR REPRODUCIBILITY
# ============================================================

review_decisions = review_decisions[
    ~review_decisions["nct_id"].isin(exclude_ids)
].copy()

new_review_rows = pd.DataFrame({
    "nct_id": list(exclude_ids),
    "review_type": "PHASE46_MISSING_FOCAL_SCOPE_CHECK",
    "initial_category": "POST_ROLE_AUDIT",
    "final_decision": "EXCLUDE",
    "decision_confidence": "high",
    "decision_reason": [reason_map[nct] for nct in exclude_ids],
    "source_url": [source_map[nct] for nct in exclude_ids],
})

review_decisions = pd.concat(
    [review_decisions, new_review_rows],
    ignore_index=True
)

if review_decisions["nct_id"].duplicated().any():
    dupes = review_decisions.loc[
        review_decisions["nct_id"].duplicated(keep=False),
        "nct_id"
    ].tolist()

    raise ValueError(
        f"Duplicate NCT IDs in review_decisions_final.csv after patch: {dupes}"
    )

review_decisions.to_csv(
    review_decisions_file,
    index=False
)


# ============================================================
# 5. REVISED INCLUDED TRIAL IDS
# ============================================================

included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

if len(included_ids) != 1003:
    raise ValueError(
        f"Expected 1003 included trials after final scope correction, "
        f"found {len(included_ids)}."
    )


# ============================================================
# 6. FILTER CURRENT CLEAN RELATIONAL TABLES
# ============================================================

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


# ============================================================
# 7. MARK ALL ROWS OF EXCLUDED TRIALS OUT OF SCOPE
# ============================================================

roles.loc[
    roles["nct_id"].isin(exclude_ids),
    "final_role"
] = "OUT_OF_SCOPE"

roles.loc[
    roles["nct_id"].isin(exclude_ids),
    "role_review_confidence_final"
] = "high"

roles.loc[
    roles["nct_id"].isin(exclude_ids),
    "role_review_reason_final"
] = roles.loc[
    roles["nct_id"].isin(exclude_ids),
    "nct_id"
].map(reason_map)


# ============================================================
# 8. ASSIGN THE FIVE MISSING FOCAL THERAPIES
# ============================================================

def searchable_row_text(row):
    columns = [
        "canonical_name_final",
        "analysis_name_candidate",
        "canonical_candidate",
        "active_name_key",
        "active_component_raw",
        "active_component_source",
        "intervention_name",
    ]

    values = []

    for column in columns:
        if column in row.index and pd.notna(row[column]):
            values.append(str(row[column]).lower())

    return " ".join(values)


for nct_id, therapy in focal_map.items():

    trial_rows = roles[
        roles["nct_id"] == nct_id
    ].copy()

    if trial_rows.empty:
        raise ValueError(
            f"No intervention-role rows found for focal correction {nct_id}."
        )

    therapy_lower = therapy.lower()

    match_indices = []

    for idx, row in trial_rows.iterrows():
        if therapy_lower in searchable_row_text(row):
            match_indices.append(idx)

    if not match_indices:
        raise ValueError(
            f"Could not locate a {therapy} intervention row for {nct_id}.\n"
            + trial_rows[
                [
                    column
                    for column in [
                        "nct_id",
                        "intervention_name",
                        "active_component_source",
                        "analysis_name_candidate",
                        "canonical_name_final",
                        "final_role"
                    ]
                    if column in trial_rows.columns
                ]
            ].to_string(index=False)
        )

    # All rows representing the focal therapy are marked focal.
    roles.loc[
        match_indices,
        "final_role"
    ] = "FOCAL_OBESITY_THERAPY"

    roles.loc[
        match_indices,
        "role_review_confidence_final"
    ] = "high"

    roles.loc[
        match_indices,
        "role_review_reason_final"
    ] = reason_map[nct_id]

    roles.loc[
        match_indices,
        "canonical_name_final"
    ] = therapy

    print(
        f"Assigned focal therapy for {nct_id}: "
        f"{therapy} ({len(match_indices)} row(s))"
    )


# ============================================================
# 9. SAVE FINAL ROLE TABLE
# ============================================================

roles.to_csv(
    roles_file,
    index=False
)


# ============================================================
# 10. REGENERATE ANALYSIS-READY INTERVENTION TABLE
# ============================================================

analysis_ready = roles[
    roles["nct_id"].isin(included_ids)
    & ~roles["final_role"].isin(
        [
            "CONTROL_PLACEBO",
            "OUT_OF_SCOPE"
        ]
    )
].copy()

analysis_ready.to_csv(
    analysis_ready_file,
    index=False
)


# ============================================================
# 11. REGENERATE FOCAL THERAPY MAP
# ============================================================

focal = analysis_ready[
    analysis_ready["final_role"]
    == "FOCAL_OBESITY_THERAPY"
].copy()

focal_map_df = (
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
        subset=[
            "nct_id",
            "canonical_name_final"
        ]
    )
    .sort_values(
        [
            "nct_id",
            "canonical_name_final"
        ]
    )
)

focal_map_df.to_csv(
    focal_map_file,
    index=False
)


# ============================================================
# 12. FINAL VALIDATION
# ============================================================

represented_ids = set(
    focal_map_df["nct_id"]
)

missing_focal_ids = (
    included_ids
    - represented_ids
)

print()
print("Final trial decisions:")
print(
    trial_decisions[
        "final_decision"
    ].value_counts()
)

print()
print("Final intervention roles:")
print(
    roles[
        "final_role"
    ].value_counts()
)

print()
print("=== Final clean dataset sizes ===")
print("Trials:", len(trials))
print("Interventions:", len(interventions))
print("Locations:", len(locations))
print("Collaborators:", len(collaborators))

print()
print(
    "Analysis-ready intervention rows:",
    len(analysis_ready)
)

print(
    "Focal therapy rows:",
    len(focal)
)

print(
    "Unique trial × focal-therapy pairs:",
    len(focal_map_df)
)

print(
    "Trials represented in focal-therapy map:",
    focal_map_df["nct_id"].nunique()
)

print(
    "Included trials with no focal therapy:",
    len(missing_focal_ids)
)

if missing_focal_ids:
    print(sorted(missing_focal_ids))

if missing_focal_ids:
    raise ValueError(
        "Final focal-therapy validation failed."
    )

print()
print("PHASE 4.6 COMPLETE: every included trial has at least one focal therapy.")
