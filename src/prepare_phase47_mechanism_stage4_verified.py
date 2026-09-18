from pathlib import Path
import pandas as pd

processed = Path("data/processed")

focal_map_file = processed / "obesity_trial_focal_therapy_map.csv"
seed_file = processed / "phase47_mechanism_seed_stage4_verified.csv"

focal = pd.read_csv(focal_map_file)
seed = pd.read_csv(seed_file)

print("Mechanism seed file:", seed_file)
print("Mechanism seed rows:", len(seed))

expected_stage4_seed = {
    "syh2082": "VERIFIED",
    "syntocinon": "VERIFIED_ALIAS",
    "thdbh120": "VERIFIED",
    "zp8396": "VERIFIED_ALIAS",
    "zt006": "VERIFIED",
    "RM-718": "VERIFIED",
    "apitegromab": "VERIFIED",
    "aro-alk7": "VERIFIED",
    "aro-inhbe": "VERIFIED",
}

seed_status_lookup = (
    seed.set_index("source_therapy_name")["classification_status"]
    .to_dict()
)

seed_errors = []

for therapy_name, expected_status in expected_stage4_seed.items():
    actual_status = seed_status_lookup.get(therapy_name)

    if actual_status != expected_status:
        seed_errors.append(
            f"{therapy_name}: expected {expected_status}, found {actual_status}"
        )

if len(seed) != 134:
    seed_errors.append(
        f"expected exactly 134 seed rows, found {len(seed)}"
    )

if seed_errors:
    raise ValueError(
        "Wrong or stale Stage-4 mechanism seed loaded:\n- "
        + "\n- ".join(seed_errors)
    )

print("Stage-4 seed sanity check: PASSED")

print("=== Phase 4.7: mechanism / target / modality stage 4 ===")
print("Trial × focal-therapy pairs:", len(focal))
print("Trials:", focal["nct_id"].nunique())
print("Source therapy labels:", focal["canonical_name_final"].nunique())


# ============================================================
# 1. VALIDATE SEED
# ============================================================

if seed["source_therapy_name"].duplicated().any():
    dupes = seed.loc[
        seed["source_therapy_name"].duplicated(keep=False),
        "source_therapy_name"
    ].tolist()
    raise ValueError(f"Duplicate source therapy names in seed: {dupes}")

source_names = set(focal["canonical_name_final"].dropna())
seed_names = set(seed["source_therapy_name"].dropna())

not_in_focal = seed_names - source_names

if not_in_focal:
    print()
    print("Seed names not present in focal map:")
    print(sorted(not_in_focal))


# ============================================================
# 2. BUILD UNIQUE THERAPY REVIEW TABLE
# ============================================================

trial_counts = (
    focal.groupby("canonical_name_final")["nct_id"]
    .nunique()
    .rename("trial_count")
    .reset_index()
)

row_counts = (
    focal.groupby("canonical_name_final")
    .size()
    .rename("pair_count")
    .reset_index()
)

review = trial_counts.merge(
    row_counts,
    on="canonical_name_final",
    how="left"
)

review = review.sort_values(
    ["trial_count", "canonical_name_final"],
    ascending=[False, True]
).reset_index(drop=True)

review["frequency_rank"] = range(1, len(review) + 1)

review = review.merge(
    seed,
    left_on="canonical_name_final",
    right_on="source_therapy_name",
    how="left"
)

review["classification_status"] = review["classification_status"].fillna(
    "UNMAPPED"
)

review["therapy_identity_final"] = review["therapy_identity_final"].fillna(
    review["canonical_name_final"]
)

review["needs_mechanism_review"] = ~review["classification_status"].isin(
    ["VERIFIED", "VERIFIED_ALIAS"]
)


# ============================================================
# 3. MERGE ONTO ALL TRIAL × THERAPY PAIRS
# ============================================================

mapping_cols = [
    "source_therapy_name",
    "therapy_identity_final",
    "mechanism_superfamily",
    "mechanism_family",
    "molecular_targets",
    "mechanism_action",
    "modality",
    "agonist_order",
    "incretin_based",
    "classification_status",
    "confidence",
    "source_url",
    "notes",
    "scope_review_flag",
]

stage1 = focal.merge(
    seed[mapping_cols],
    left_on="canonical_name_final",
    right_on="source_therapy_name",
    how="left"
)

stage1["classification_status"] = stage1["classification_status"].fillna(
    "UNMAPPED"
)

stage1["therapy_identity_final"] = stage1["therapy_identity_final"].fillna(
    stage1["canonical_name_final"]
)

def parse_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}

stage1["scope_review_flag"] = (
    stage1["scope_review_flag"].map(parse_bool)
)


# ============================================================
# 4. OUTPUTS
# ============================================================

review_file = processed / "phase47_mechanism_review.csv"
stage1_file = processed / "phase47_focal_therapy_mechanism_stage1.csv"
unmapped_file = processed / "phase47_unmapped_therapies.csv"
scope_flag_file = processed / "phase47_scope_review_flags.csv"

review.to_csv(review_file, index=False)
stage1.to_csv(stage1_file, index=False)

unmapped = review[
    review["classification_status"].isin(
        ["UNMAPPED", "RESEARCH_REQUIRED"]
    )
].copy()

unmapped.to_csv(unmapped_file, index=False)

scope_flags = review[
    review["scope_review_flag"].map(parse_bool)
].copy()

scope_flags.to_csv(scope_flag_file, index=False)


# ============================================================
# 5. COVERAGE SUMMARY
# ============================================================

fully_classified = stage1[
    stage1["classification_status"].isin(
        ["VERIFIED", "VERIFIED_ALIAS"]
    )
]

seed_listed = stage1[
    stage1["classification_status"] != "UNMAPPED"
]

print()
print("=== Stage-1 classification coverage ===")
print(
    "Unique source therapy labels fully classified:",
    review["classification_status"]
    .isin(["VERIFIED", "VERIFIED_ALIAS"])
    .sum()
)

print(
    "Unique source therapy labels listed but mechanism unresolved:",
    (review["classification_status"] == "RESEARCH_REQUIRED").sum()
)

print(
    "Unique source therapy labels still unmapped:",
    (review["classification_status"] == "UNMAPPED").sum()
)

print(
    "Trial × therapy pairs fully classified:",
    len(fully_classified),
    f"({len(fully_classified) / len(stage1):.1%})"
)

print(
    "Trial × therapy pairs represented in seed:",
    len(seed_listed),
    f"({len(seed_listed) / len(stage1):.1%})"
)

print()
print(
    "Unique therapy identities after known alias collapse:",
    stage1["therapy_identity_final"].nunique()
)

print()
print("=== Classified pairs by mechanism superfamily ===")
print(
    fully_classified["mechanism_superfamily"]
    .value_counts(dropna=False)
)

print()
print("=== Top 30 remaining therapies requiring work ===")
print(
    unmapped[
        [
            "frequency_rank",
            "canonical_name_final",
            "trial_count",
            "classification_status",
        ]
    ]
    .head(30)
    .to_string(index=False)
)

print()
print("=== Scope-review flags ===")
if len(scope_flags) == 0:
    print("None")
else:
    print(
        scope_flags[
            [
                "canonical_name_final",
                "trial_count",
                "notes",
            ]
        ].to_string(index=False)
    )

print()
print("Saved:")
print(review_file)
print(stage1_file)
print(unmapped_file)
print(scope_flag_file)
