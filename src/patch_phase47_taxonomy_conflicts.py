from pathlib import Path
import pandas as pd

processed = Path("data/processed")
seed_file = processed / "phase47_mechanism_seed_stage10_verified.csv"

seed = pd.read_csv(seed_file)

print("=== Phase 4.7 taxonomy conflict patch ===")
print("Seed rows:", len(seed))

# ------------------------------------------------------------
# 1. Semaglutide
#    Modality should describe the therapeutic molecule, not the
#    delivery device/formulation.
# ------------------------------------------------------------

semaglutide_labels = {
    "semaglutide",
    "dwc202502",
    "dwc202503",
    "dwrx5003",
}

semaglutide_mask = (
    seed["therapy_identity_final"]
    .fillna("")
    .astype(str)
    .eq("semaglutide")
)

seed.loc[
    semaglutide_mask,
    "modality"
] = "peptide"

print(
    "Semaglutide rows normalised:",
    int(semaglutide_mask.sum())
)

# ------------------------------------------------------------
# 2. Exenatide
#    Implant/microsphere/once-weekly wording is formulation,
#    not a different molecular modality.
# ------------------------------------------------------------

exenatide_mask = (
    seed["therapy_identity_final"]
    .fillna("")
    .astype(str)
    .eq("exenatide")
)

seed.loc[
    exenatide_mask,
    "modality"
] = "peptide"

print(
    "Exenatide rows normalised:",
    int(exenatide_mask.sum())
)

# ------------------------------------------------------------
# 3. EMP16
#    Normalise ASCII/Greek alpha character in target text.
# ------------------------------------------------------------

emp16_mask = (
    seed["therapy_identity_final"]
    .fillna("")
    .astype(str)
    .eq("EMP16 (orlistat + acarbose)")
)

seed.loc[
    emp16_mask,
    "molecular_targets"
] = (
    "gastric/pancreatic lipases; "
    "intestinal alpha-glucosidases/amylase"
)

print(
    "EMP16 rows normalised:",
    int(emp16_mask.sum())
)

# ------------------------------------------------------------
# 4. Validation
# ------------------------------------------------------------

def unique_nonblank(mask, column):
    vals = (
        seed.loc[mask, column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted(set(v for v in vals if v))

checks = {
    "semaglutide modality": unique_nonblank(
        semaglutide_mask, "modality"
    ),
    "exenatide modality": unique_nonblank(
        exenatide_mask, "modality"
    ),
    "EMP16 molecular_targets": unique_nonblank(
        emp16_mask, "molecular_targets"
    ),
}

print()
print("=== Validation ===")
for label, values in checks.items():
    print(f"{label}: {values}")
    if len(values) != 1:
        raise ValueError(
            f"{label} still has conflicting values: {values}"
        )

seed.to_csv(seed_file, index=False)

print()
print("Patched:")
print(seed_file)
print()
print(
    "Conflict patch complete. Re-run Stage 10, then re-run "
    "finalize_phase47_taxonomy.py."
)
