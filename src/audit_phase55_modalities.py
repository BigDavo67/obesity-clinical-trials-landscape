from pathlib import Path
import pandas as pd

processed = Path("data/processed")

pair_file = processed / "phase5_trial_therapy_master.csv"
audit_out = processed / "phase55_modality_audit.csv"

pair = pd.read_csv(pair_file)

print("=== Phase 5.5A: modality audit ===")
print("Input trial × focal-therapy pairs:", len(pair))

required = {
    "nct_id",
    "therapy_identity_final",
    "classification_status_final",
    "mechanism_superfamily_final",
    "modality",
    "start_year",
    "start_timing_vs_snapshot",
}
missing = required - set(pair.columns)
if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )

verified = pair[
    pair["classification_status_final"].isin(
        ["VERIFIED", "VERIFIED_ALIAS"]
    )
].copy()

print("Verified mechanism pairs:", len(verified))

if len(verified) != 972:
    raise ValueError(
        f"Expected 972 verified pairs; found {len(verified)}."
    )

# ------------------------------------------------------------
# 1. Raw modality frequencies
# ------------------------------------------------------------

raw = (
    verified.groupby("modality", dropna=False)
    .agg(
        pair_count=("nct_id", "size"),
        unique_trials=("nct_id", "nunique"),
        unique_identities=("therapy_identity_final", "nunique"),
    )
    .reset_index()
    .sort_values(
        ["pair_count", "unique_identities"],
        ascending=[False, False],
    )
)

raw["pair_share"] = (
    raw["pair_count"] / len(verified)
)

# ------------------------------------------------------------
# 2. Recent vs historical raw modality presence
# ------------------------------------------------------------

verified["era"] = "2015–2022"

recent_mask = (
    verified["start_year"].between(2023, 2025)
    | (
        verified["start_year"].eq(2026)
        & verified["start_timing_vs_snapshot"].eq(
            "BEFORE_SNAPSHOT"
        )
    )
)

verified.loc[
    recent_mask,
    "era"
] = "2023–2026 YTD"

era_counts = (
    verified.groupby(["modality", "era"], dropna=False)
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for col in ["2015–2022", "2023–2026 YTD"]:
    if col not in era_counts.columns:
        era_counts[col] = 0

raw = raw.merge(
    era_counts,
    on="modality",
    how="left",
)

# ------------------------------------------------------------
# 3. Candidate high-level modality group
#    This is an audit proposal only; no final grouping is frozen yet.
# ------------------------------------------------------------

def candidate_group(value):
    if pd.isna(value):
        return "Unknown / unspecified"

    s = str(value).strip().lower()

    if "sirna" in s or "rnai" in s or "antisense" in s or "oligonucleotide" in s:
        return "RNA / oligonucleotide"

    if "antibody" in s or "ligand trap" in s:
        return "Antibody / biologic"

    if "fc-fusion" in s or "fc fusion" in s:
        return "Protein / peptide biologic"

    if "peptide" in s or "protein" in s or "cytokine" in s:
        if "small molecule" in s:
            return "Combination / mixed modality"
        return "Peptide / protein"

    if "small molecule" in s:
        return "Small molecule"

    if "polymer" in s:
        return "Polymer / gut-restricted"

    if "combination" in s or "mixed" in s:
        return "Combination / mixed modality"

    if "unspecified" in s or "unknown" in s or s == "":
        return "Unknown / unspecified"

    return "Other"

raw["candidate_modality_group"] = (
    raw["modality"].map(candidate_group)
)

raw.to_csv(audit_out, index=False)

# ------------------------------------------------------------
# 4. Print useful output
# ------------------------------------------------------------

print()
print("Unique raw modality labels:")
print(raw["modality"].nunique(dropna=False))

print()
print("=== Raw modality labels by pair count ===")
print(
    raw[
        [
            "modality",
            "pair_count",
            "pair_share",
            "unique_identities",
            "2015–2022",
            "2023–2026 YTD",
            "candidate_modality_group",
        ]
    ]
    .head(60)
    .to_string(
        index=False,
        formatters={
            "pair_share": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("=== Candidate modality-group totals ===")
candidate_summary = (
    raw.groupby("candidate_modality_group")
    .agg(
        pair_count=("pair_count", "sum"),
        unique_raw_labels=("modality", "nunique"),
        unique_identities=("unique_identities", "sum"),
    )
    .sort_values("pair_count", ascending=False)
)

print(candidate_summary.to_string())

print()
print("Saved:")
print(audit_out)

print()
print("PHASE 5.5A MODALITY AUDIT COMPLETE.")
