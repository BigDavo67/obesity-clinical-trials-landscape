from pathlib import Path
import pandas as pd

processed = Path("data/processed")

pair_file = processed / "phase47_focal_therapy_mechanism_stage1.csv"
review_file = processed / "phase47_mechanism_review.csv"

pairs = pd.read_csv(pair_file)
review = pd.read_csv(review_file)

print("=== Phase 4.7 final taxonomy freeze ===")
print("Input trial × focal-therapy pairs:", len(pairs))
print("Input source therapy labels:", len(review))

required_pair_cols = {
    "nct_id",
    "canonical_name_final",
    "therapy_identity_final",
    "classification_status",
    "mechanism_superfamily",
    "mechanism_family",
    "molecular_targets",
    "mechanism_action",
    "modality",
    "agonist_order",
    "incretin_based",
}
missing = required_pair_cols - set(pairs.columns)
if missing:
    raise ValueError(
        f"Missing required columns from pair file: {sorted(missing)}"
    )

# ------------------------------------------------------------
# 1. DEFINE FINAL CLASSIFICATION STATES
# ------------------------------------------------------------

verified_statuses = {"VERIFIED", "VERIFIED_ALIAS"}

def final_status(status):
    if status in verified_statuses:
        return status
    if status == "RESEARCH_REQUIRED":
        return "UNRESOLVED_PUBLIC_INFO"
    if status == "UNMAPPED":
        return "UNCLASSIFIED_LONG_TAIL"
    return status

def final_superfamily(row):
    status = row["classification_status"]
    if status in verified_statuses:
        return row["mechanism_superfamily"]
    if status == "RESEARCH_REQUIRED":
        return "Undisclosed / insufficient public information"
    if status == "UNMAPPED":
        return "Unclassified long tail"
    return row["mechanism_superfamily"]

def final_family(row):
    status = row["classification_status"]
    if status in verified_statuses:
        return row["mechanism_family"]
    if status == "RESEARCH_REQUIRED":
        return "Undisclosed"
    if status == "UNMAPPED":
        return "Not individually reviewed"
    return row["mechanism_family"]

pairs["classification_status_original"] = pairs["classification_status"]
pairs["classification_status_final"] = (
    pairs["classification_status"]
    .map(final_status)
)

pairs["mechanism_superfamily_final"] = pairs.apply(
    final_superfamily,
    axis=1,
)

pairs["mechanism_family_final"] = pairs.apply(
    final_family,
    axis=1,
)

pairs["mechanism_classified"] = (
    pairs["classification_status_original"]
    .isin(verified_statuses)
)

pairs["researched_but_publicly_undisclosed"] = (
    pairs["classification_status_original"]
    .eq("RESEARCH_REQUIRED")
)

pairs["unreviewed_long_tail"] = (
    pairs["classification_status_original"]
    .eq("UNMAPPED")
)

# Do NOT invent target/action/modality for unresolved assets.
unresolved_mask = ~pairs["mechanism_classified"]

for col in [
    "molecular_targets",
    "mechanism_action",
    "modality",
    "agonist_order",
    "incretin_based",
]:
    # Preserve any pre-existing value for RESEARCH_REQUIRED if present,
    # but leave true UNMAPPED rows explicitly blank.
    pairs.loc[
        pairs["classification_status_original"].eq("UNMAPPED"),
        col,
    ] = pd.NA

# ------------------------------------------------------------
# 2. VALIDATE PAIR-LEVEL POPULATION
# ------------------------------------------------------------

pair_duplicates = pairs.duplicated(
    subset=["nct_id", "canonical_name_final"],
    keep=False,
)

if pair_duplicates.any():
    dupes = pairs.loc[
        pair_duplicates,
        ["nct_id", "canonical_name_final"],
    ]
    raise ValueError(
        "Duplicate trial × source-therapy pairs found:\n"
        + dupes.to_string(index=False)
    )

status_counts = (
    pairs["classification_status_final"]
    .value_counts()
)

classified_pairs = int(pairs["mechanism_classified"].sum())
researched_unknown_pairs = int(
    pairs["researched_but_publicly_undisclosed"].sum()
)
long_tail_pairs = int(pairs["unreviewed_long_tail"].sum())

print()
print("=== Final pair-level classification states ===")
print(status_counts)
print()
print(
    "Verified mechanism pairs:",
    classified_pairs,
    f"({classified_pairs / len(pairs):.1%})",
)
print(
    "Researched but public mechanism unresolved:",
    researched_unknown_pairs,
    f"({researched_unknown_pairs / len(pairs):.1%})",
)
print(
    "Unreviewed long-tail pairs:",
    long_tail_pairs,
    f"({long_tail_pairs / len(pairs):.1%})",
)

# ------------------------------------------------------------
# 3. FINAL UNIQUE SOURCE-LABEL TAXONOMY
# ------------------------------------------------------------

review["classification_status_original"] = (
    review["classification_status"]
)

review["classification_status_final"] = (
    review["classification_status"]
    .map(final_status)
)

review["mechanism_superfamily_final"] = review.apply(
    final_superfamily,
    axis=1,
)

review["mechanism_family_final"] = review.apply(
    final_family,
    axis=1,
)

review["mechanism_classified"] = (
    review["classification_status_original"]
    .isin(verified_statuses)
)

review["researched_but_publicly_undisclosed"] = (
    review["classification_status_original"]
    .eq("RESEARCH_REQUIRED")
)

review["unreviewed_long_tail"] = (
    review["classification_status_original"]
    .eq("UNMAPPED")
)

# ------------------------------------------------------------
# 4. CANONICAL IDENTITY-LEVEL TAXONOMY
# ------------------------------------------------------------

# Use pair data so counts reflect the final analysis population.
identity_rows = []

for identity, group in pairs.groupby(
    "therapy_identity_final",
    dropna=False,
):
    verified = group[
        group["mechanism_classified"]
    ].copy()

    if not verified.empty:
        # Detect contradictory mappings among aliases collapsed to one identity.
        consistency_cols = [
            "mechanism_superfamily_final",
            "mechanism_family_final",
            "molecular_targets",
            "modality",
            "agonist_order",
        ]

        conflicts = {}
        for col in consistency_cols:
            vals = sorted(
                {
                    str(v)
                    for v in verified[col].dropna()
                    if str(v).strip() != ""
                }
            )
            if len(vals) > 1:
                conflicts[col] = " | ".join(vals)

        first = verified.iloc[0]

        identity_rows.append({
            "therapy_identity_final": identity,
            "trial_count": group["nct_id"].nunique(),
            "pair_count": len(group),
            "source_label_count": group[
                "canonical_name_final"
            ].nunique(),
            "mechanism_superfamily_final": (
                first["mechanism_superfamily_final"]
            ),
            "mechanism_family_final": (
                first["mechanism_family_final"]
            ),
            "molecular_targets": first["molecular_targets"],
            "mechanism_action": first["mechanism_action"],
            "modality": first["modality"],
            "agonist_order": first["agonist_order"],
            "incretin_based": first["incretin_based"],
            "identity_classification_state": "VERIFIED",
            "has_mapping_conflict": bool(conflicts),
            "mapping_conflict_detail": (
                "; ".join(
                    f"{k}: {v}"
                    for k, v in conflicts.items()
                )
                if conflicts
                else ""
            ),
        })

    else:
        # No verified mechanism for this identity.
        has_researched_unknown = (
            group[
                "researched_but_publicly_undisclosed"
            ].any()
        )

        if has_researched_unknown:
            state = "UNRESOLVED_PUBLIC_INFO"
            superfam = (
                "Undisclosed / insufficient public information"
            )
            family = "Undisclosed"
        else:
            state = "UNCLASSIFIED_LONG_TAIL"
            superfam = "Unclassified long tail"
            family = "Not individually reviewed"

        identity_rows.append({
            "therapy_identity_final": identity,
            "trial_count": group["nct_id"].nunique(),
            "pair_count": len(group),
            "source_label_count": group[
                "canonical_name_final"
            ].nunique(),
            "mechanism_superfamily_final": superfam,
            "mechanism_family_final": family,
            "molecular_targets": pd.NA,
            "mechanism_action": pd.NA,
            "modality": pd.NA,
            "agonist_order": pd.NA,
            "incretin_based": pd.NA,
            "identity_classification_state": state,
            "has_mapping_conflict": False,
            "mapping_conflict_detail": "",
        })

identity = pd.DataFrame(identity_rows)

identity = identity.sort_values(
    ["trial_count", "therapy_identity_final"],
    ascending=[False, True],
).reset_index(drop=True)

conflicts = identity[
    identity["has_mapping_conflict"]
].copy()

# ------------------------------------------------------------
# 5. DASHBOARD-SAFE MECHANISM DISTRIBUTION
# ------------------------------------------------------------

mechanism_summary = (
    pairs.groupby(
        [
            "mechanism_superfamily_final",
            "classification_status_final",
        ],
        dropna=False,
    )
    .agg(
        pair_count=("nct_id", "size"),
        trial_count=("nct_id", "nunique"),
    )
    .reset_index()
)

mechanism_summary["pair_share"] = (
    mechanism_summary["pair_count"]
    / len(pairs)
)

mechanism_summary = mechanism_summary.sort_values(
    ["pair_count", "mechanism_superfamily_final"],
    ascending=[False, True],
)

# ------------------------------------------------------------
# 6. SAVE FINAL OUTPUTS
# ------------------------------------------------------------

pairs_out = (
    processed
    / "obesity_focal_therapy_taxonomy_final.csv"
)

labels_out = (
    processed
    / "therapy_source_label_taxonomy_final.csv"
)

identity_out = (
    processed
    / "therapy_identity_taxonomy_final.csv"
)

summary_out = (
    processed
    / "mechanism_superfamily_summary_final.csv"
)

conflicts_out = (
    processed
    / "phase47_taxonomy_conflicts.csv"
)

pairs.to_csv(
    pairs_out,
    index=False,
)

review.to_csv(
    labels_out,
    index=False,
)

identity.to_csv(
    identity_out,
    index=False,
)

mechanism_summary.to_csv(
    summary_out,
    index=False,
)

conflicts.to_csv(
    conflicts_out,
    index=False,
)

print()
print("=== Identity-level taxonomy ===")
print(
    identity["identity_classification_state"]
    .value_counts()
)

print(
    "Unique canonical therapy identities:",
    len(identity),
)

print(
    "Verified identities:",
    int(
        (
            identity["identity_classification_state"]
            == "VERIFIED"
        ).sum()
    ),
)

print(
    "Researched but unresolved identities:",
    int(
        (
            identity["identity_classification_state"]
            == "UNRESOLVED_PUBLIC_INFO"
        ).sum()
    ),
)

print(
    "Unreviewed long-tail identities:",
    int(
        (
            identity["identity_classification_state"]
            == "UNCLASSIFIED_LONG_TAIL"
        ).sum()
    ),
)

print(
    "Identity mapping conflicts:",
    len(conflicts),
)

print()
print("Saved final Phase 4.7 outputs:")
print(pairs_out)
print(labels_out)
print(identity_out)
print(summary_out)
print(conflicts_out)

print()
print(
    "PHASE 4.7 TAXONOMY FREEZE COMPLETE."
)
