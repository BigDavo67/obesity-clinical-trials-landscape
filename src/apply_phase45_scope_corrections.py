from pathlib import Path
import pandas as pd


# ============================================================
# PHASE 4.5 SCOPE CORRECTIONS
# ============================================================

processed_data_dir = Path("data/processed")

review_file = processed_data_dir / "review_decisions_final.csv"
triage_file = processed_data_dir / "trial_inclusion_triage.csv"
correction_log_file = processed_data_dir / "phase45_scope_corrections.csv"

review_df = pd.read_csv(review_file)
triage_df = pd.read_csv(triage_file)

corrections = {
    "NCT04614233": (
        "EXCLUDE",
        "Phase 4.5 active-intervention audit found that the active intervention "
        "RiduZone (90% OEA) is registered as DIETARY_SUPPLEMENT; the only DRUG "
        "intervention is placebo. This falls outside the defined pharmacotherapy scope."
    ),
    "NCT06512818": (
        "EXCLUDE",
        "Phase 4.5 active-intervention audit found that the active intervention "
        "Psiguavin is registered as DIETARY_SUPPLEMENT; the only DRUG intervention "
        "is placebo. This falls outside the defined pharmacotherapy scope."
    ),
}

category_col = (
    "auto_review_category"
    if "auto_review_category" in triage_df.columns
    else "review_category"
)

triage_lookup = (
    triage_df
    .set_index("nct_id")
)

correction_rows = []

for nct_id, (decision, reason) in corrections.items():

    if nct_id not in triage_lookup.index:
        raise ValueError(
            f"{nct_id} was not found in trial_inclusion_triage.csv."
        )

    initial_category = triage_lookup.loc[
        nct_id,
        category_col
    ]

    correction_rows.append(
        {
            "nct_id": nct_id,
            "review_type": "PHASE45_SCOPE_CHECK",
            "initial_category": initial_category,
            "final_decision": decision,
            "decision_confidence": "high",
            "decision_reason": reason,
            "source_url": f"https://clinicaltrials.gov/study/{nct_id}",
        }
    )

correction_df = pd.DataFrame(correction_rows)

# Remove any older decision rows for these IDs, then append the corrected rows.
review_df = review_df[
    ~review_df["nct_id"].isin(corrections.keys())
].copy()

review_df = pd.concat(
    [
        review_df,
        correction_df
    ],
    ignore_index=True
)

if review_df["nct_id"].duplicated().any():
    duplicates = review_df.loc[
        review_df["nct_id"].duplicated(keep=False),
        "nct_id"
    ].tolist()

    raise ValueError(
        f"Duplicate NCT IDs remain after correction: {duplicates}"
    )

review_df.to_csv(
    review_file,
    index=False
)

correction_df.to_csv(
    correction_log_file,
    index=False
)

print("=== Phase 4.5 scope corrections applied ===")
print(correction_df[[
    "nct_id",
    "final_decision",
    "decision_reason"
]].to_string(index=False))

print()
print("Updated reviewed decision rows:", len(review_df))
print("Saved:", review_file)
print("Correction log:", correction_log_file)

print()
print(
    "Next: rerun python src/finalize_trial_inclusion.py "
    "to propagate these two exclusions."
)
