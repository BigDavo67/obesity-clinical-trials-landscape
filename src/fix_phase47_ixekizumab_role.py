from pathlib import Path
import pandas as pd

processed = Path("data/processed")

roles_file = processed / "obesity_intervention_roles_final.csv"
analysis_ready_file = processed / "obesity_interventions_analysis_ready.csv"
focal_map_file = processed / "obesity_trial_focal_therapy_map.csv"
trial_decisions_file = processed / "trial_decisions_final.csv"

roles = pd.read_csv(roles_file)
trial_decisions = pd.read_csv(trial_decisions_file)

target_trials = {
    "NCT06588283": (
        "Ixekizumab treats plaque psoriasis; tirzepatide is the "
        "weight-loss/obesity therapy in the TOGETHER-PsO trial."
    ),
    "NCT06588296": (
        "Ixekizumab treats psoriatic arthritis; tirzepatide is the "
        "weight-loss/obesity therapy in the TOGETHER-PsA trial."
    ),
    "NCT07443956": (
        "Ixekizumab is the anti-inflammatory PsA therapy; tirzepatide "
        "is the anti-obesity/weight-loss therapy in COMBAT-PsA."
    ),
}

target_ids = set(target_trials)

print("=== Phase 4.7 pre-classification role correction ===")

# Require all three trials to remain included.
included_ids = set(
    trial_decisions.loc[
        trial_decisions["final_decision"] == "INCLUDE",
        "nct_id"
    ]
)

missing_included = target_ids - included_ids
if missing_included:
    raise ValueError(
        f"Expected ixekizumab trials are not all included: "
        f"{sorted(missing_included)}"
    )

# Find ixekizumab role rows robustly using all available name fields.
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

ix_mask = roles["nct_id"].isin(target_ids)

name_mask = False
for col in name_cols:
    current = (
        roles[col]
        .fillna("")
        .astype(str)
        .str.contains("ixekizumab", case=False, regex=False)
    )
    name_mask = current if isinstance(name_mask, bool) else (name_mask | current)

ix_mask = ix_mask & name_mask

ix_rows = roles.loc[ix_mask].copy()

print("Ixekizumab rows found:", len(ix_rows))
print(
    ix_rows[
        [
            col for col in [
                "nct_id",
                "canonical_name_final",
                "analysis_name_candidate",
                "final_role"
            ]
            if col in ix_rows.columns
        ]
    ].to_string(index=False)
)

if ix_rows["nct_id"].nunique() != 3:
    raise ValueError(
        "Expected ixekizumab rows in exactly three trials."
    )

# Only alter the role; do not exclude the trials.
roles.loc[ix_mask, "final_role"] = "BACKGROUND_OR_CONCOMITANT"

if "role_review_confidence_final" in roles.columns:
    roles.loc[ix_mask, "role_review_confidence_final"] = "high"

if "role_review_reason_final" in roles.columns:
    roles.loc[ix_mask, "role_review_reason_final"] = (
        roles.loc[ix_mask, "nct_id"]
        .map(target_trials)
    )

# Ensure tirzepatide remains a focal therapy in all three trials.
def row_contains_therapy(row, therapy):
    for col in name_cols:
        value = row.get(col, "")
        if pd.notna(value) and therapy.lower() in str(value).lower():
            return True
    return False

for nct_id in sorted(target_ids):
    subset = roles[roles["nct_id"] == nct_id]

    tirz_focal = subset[
        subset.apply(
            lambda row: row_contains_therapy(row, "tirzepatide"),
            axis=1
        )
        & (subset["final_role"] == "FOCAL_OBESITY_THERAPY")
    ]

    if tirz_focal.empty:
        raise ValueError(
            f"{nct_id} no longer has focal tirzepatide."
        )

# Save corrected full role table.
roles.to_csv(roles_file, index=False)

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

# Rebuild focal therapy map.
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

focal_map.to_csv(
    focal_map_file,
    index=False
)

missing_focal = included_ids - set(focal_map["nct_id"])

print()
print("=== Corrected focal-map validation ===")
print("Included trials:", len(included_ids))
print("Focal therapy rows:", len(focal))
print("Unique trial × focal-therapy pairs:", len(focal_map))
print("Trials represented:", focal_map["nct_id"].nunique())
print("Included trials with no focal therapy:", len(missing_focal))

ix_in_focal = focal_map[
    focal_map["canonical_name_final"]
    .fillna("")
    .astype(str)
    .str.contains("ixekizumab", case=False, regex=False)
]

print("Ixekizumab focal pairs remaining:", len(ix_in_focal))

if missing_focal:
    print(sorted(missing_focal))
    raise ValueError("Focal map lost trial coverage.")

if len(ix_in_focal) != 0:
    raise ValueError("Ixekizumab still appears in focal map.")

print()
print(
    "Correction complete: trials remain included, but ixekizumab "
    "is no longer treated as an obesity therapy."
)