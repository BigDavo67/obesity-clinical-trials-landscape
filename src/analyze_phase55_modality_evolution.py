from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed = Path("data/processed")
figures = Path("outputs/figures")
figures.mkdir(parents=True, exist_ok=True)

pair_file = processed / "phase5_trial_therapy_master.csv"

mapping_out = processed / "phase55_modality_mapping_final.csv"
overall_out = processed / "phase55_modality_landscape_overall.csv"
trend_out = processed / "phase55_modality_trend_by_year.csv"
era_out = processed / "phase55_modality_recent_vs_historical.csv"
stage_out = processed / "phase55_modality_by_pipeline_stage.csv"

fig_overall = figures / "phase55_modality_landscape_overall.png"
fig_era = figures / "phase55_modality_recent_vs_historical.png"
fig_trend = figures / "phase55_modality_trend_over_time.png"

pair = pd.read_csv(pair_file)

print("=== Phase 5.5: modality evolution ===")
print("Input trial × focal-therapy pairs:", len(pair))

verified = pair[
    pair["classification_status_final"].isin(
        ["VERIFIED", "VERIFIED_ALIAS"]
    )
].copy()

if len(verified) != 972:
    raise ValueError(
        f"Expected 972 verified pairs; found {len(verified)}."
    )

# ------------------------------------------------------------
# 1. FINAL MODALITY GROUPING
# ------------------------------------------------------------

def modality_group(value):
    if pd.isna(value):
        return "Unknown / unspecified"

    s = str(value).strip().lower()

    # RNA first
    if (
        "sirna" in s
        or "rnai" in s
        or "antisense" in s
        or "oligonucleotide" in s
    ):
        return "RNA / oligonucleotide"

    # Explicit mixed/combination labels
    if (
        "combination" in s
        or "mixed" in s
    ):
        return "Combination / mixed modality"

    # Antibody / engineered protein biologics
    if (
        "antibody" in s
        or "fc-fusion" in s
        or "fc fusion" in s
        or "ligand trap" in s
        or "large-molecule biologic" in s
        or "multi-agonist biologic" in s
        or "multi-domain biologic" in s
        or "gdf15-fc" in s
    ):
        return "Antibody / protein biologic"

    # Peptides / peptide-like agents
    if (
        "peptide" in s
        or "protein toxin biologic" in s
        or "cyclic dipeptide" in s
        or "cyclic peptide" in s
        or "cytokine/peptide" in s
        or "peptidomimetic" in s
    ):
        return "Peptide-based"

    # Small molecules
    if (
        "small molecule" in s
        or "small-molecule" in s
        or "aminosterol" in s
    ):
        return "Small molecule"

    # Gut-restricted / unusual local modality
    if "polymer" in s:
        return "Gut-restricted / other"

    if "botanical cannabinoid" in s:
        return "Gut-restricted / other"

    if "unspecified" in s or s == "":
        return "Unknown / unspecified"

    return "Gut-restricted / other"

verified["modality_group_final"] = (
    verified["modality"].map(modality_group)
)

# ------------------------------------------------------------
# 2. SAVE RAW→FINAL MAPPING
# ------------------------------------------------------------

mapping = (
    verified[
        ["modality", "modality_group_final"]
    ]
    .drop_duplicates()
    .sort_values(
        ["modality_group_final", "modality"]
    )
)

mapping.to_csv(mapping_out, index=False)

# ------------------------------------------------------------
# 3. OVERALL LANDSCAPE
# ------------------------------------------------------------

overall = (
    verified.groupby("modality_group_final")
    .agg(
        verified_pair_count=("nct_id", "size"),
        unique_trials=("nct_id", "nunique"),
        unique_therapy_identities=("therapy_identity_final", "nunique"),
    )
    .reset_index()
)

overall["share_of_verified_pairs"] = (
    overall["verified_pair_count"] / len(verified)
)

overall = overall.sort_values(
    "verified_pair_count",
    ascending=False
)

overall.to_csv(overall_out, index=False)

# ------------------------------------------------------------
# 4. CONSERVATIVE TIME TREND
# ------------------------------------------------------------

verified["include_in_confirmed_trend"] = (
    (verified["start_year"] < 2026)
    | (
        verified["start_year"].eq(2026)
        & verified["start_timing_vs_snapshot"].eq(
            "BEFORE_SNAPSHOT"
        )
    )
)

trend_df = verified[
    verified["include_in_confirmed_trend"]
].copy()

trend = (
    trend_df.groupby(
        ["start_year", "modality_group_final"]
    )
    .size()
    .rename("verified_pair_count")
    .reset_index()
)

year_totals = (
    trend_df.groupby("start_year")
    .size()
    .rename("verified_pairs_in_year")
    .reset_index()
)

trend = trend.merge(
    year_totals,
    on="start_year",
    how="left",
)

trend["share_of_verified_pairs_in_year"] = (
    trend["verified_pair_count"]
    / trend["verified_pairs_in_year"]
)

trend.to_csv(trend_out, index=False)

# ------------------------------------------------------------
# 5. HISTORICAL VS RECENT
# ------------------------------------------------------------

trend_df["era"] = np.where(
    trend_df["start_year"].between(2015, 2022),
    "2015–2022",
    "2023–2026 YTD",
)

era = (
    trend_df.groupby(
        ["modality_group_final", "era"]
    )
    .size()
    .unstack(fill_value=0)
)

for col in ["2015–2022", "2023–2026 YTD"]:
    if col not in era.columns:
        era[col] = 0

era_totals = trend_df.groupby("era").size().to_dict()

era["historical_share"] = (
    era["2015–2022"]
    / era_totals.get("2015–2022", np.nan)
)

era["recent_share"] = (
    era["2023–2026 YTD"]
    / era_totals.get("2023–2026 YTD", np.nan)
)

era["share_change_pp"] = (
    (era["recent_share"] - era["historical_share"])
    * 100
)

era = era.reset_index().sort_values(
    "recent_share",
    ascending=False
)

era.to_csv(era_out, index=False)

# ------------------------------------------------------------
# 6. PIPELINE STAGE BY MODALITY
# ------------------------------------------------------------

def broad_stage(phase):
    if phase in {
        "Early Phase 1",
        "Phase 1",
        "Phase 1/2",
    }:
        return "Early development"

    if phase in {
        "Phase 2",
        "Phase 2/3",
    }:
        return "Mid development"

    if phase == "Phase 3":
        return "Late development"

    if phase == "Phase 4":
        return "Post-marketing"

    return "Not applicable / not reported"

verified["pipeline_stage"] = (
    verified["phase_group"].map(broad_stage)
)

stage = (
    verified.groupby(
        ["modality_group_final", "pipeline_stage"]
    )
    .size()
    .unstack(fill_value=0)
)

stage["verified_pair_total"] = stage.sum(axis=1)
stage = stage.sort_values(
    "verified_pair_total",
    ascending=False
)

stage.to_csv(stage_out)

# ------------------------------------------------------------
# 7. FIGURE — OVERALL
# ------------------------------------------------------------

overall_plot = overall.sort_values(
    "verified_pair_count"
)

plt.figure(figsize=(9, 6))
plt.barh(
    overall_plot["modality_group_final"],
    overall_plot["verified_pair_count"],
)
plt.xlabel("Verified trial × therapy pairs")
plt.ylabel("Modality")
plt.title("Obesity pipeline by therapeutic modality")
plt.tight_layout()
plt.savefig(
    fig_overall,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 8. FIGURE — HISTORICAL VS RECENT
# ------------------------------------------------------------

x = np.arange(len(era))
width = 0.38

plt.figure(figsize=(10, 6))
plt.bar(
    x - width / 2,
    era["historical_share"] * 100,
    width,
    label="2015–2022",
)
plt.bar(
    x + width / 2,
    era["recent_share"] * 100,
    width,
    label="2023–2026 YTD",
)

plt.xticks(
    x,
    era["modality_group_final"],
    rotation=45,
    ha="right",
)
plt.ylabel("Share of verified pairs (%)")
plt.xlabel("Modality")
plt.title("How obesity therapeutic modalities have shifted")
plt.legend()
plt.tight_layout()
plt.savefig(
    fig_era,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 9. FIGURE — TREND OVER TIME
# ------------------------------------------------------------

trend_pivot = (
    trend.pivot(
        index="start_year",
        columns="modality_group_final",
        values="share_of_verified_pairs_in_year",
    )
    .fillna(0)
)

top_modalities = (
    overall.head(5)["modality_group_final"].tolist()
)

plt.figure(figsize=(11, 7))

for col in top_modalities:
    if col in trend_pivot.columns:
        plt.plot(
            trend_pivot.index,
            trend_pivot[col] * 100,
            marker="o",
            label=col,
        )

plt.xlabel("Trial start year")
plt.ylabel("Share of verified pairs (%)")
plt.title("Modality mix of obesity pharmacotherapy development over time")
plt.legend()
plt.tight_layout()
plt.savefig(
    fig_trend,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 10. PRINT SUMMARY
# ------------------------------------------------------------

print()
print("=== Final modality mapping ===")
print(mapping.to_string(index=False))

print()
print("=== Overall modality landscape ===")
print(
    overall[
        [
            "modality_group_final",
            "verified_pair_count",
            "share_of_verified_pairs",
            "unique_therapy_identities",
        ]
    ].to_string(
        index=False,
        formatters={
            "share_of_verified_pairs": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("=== Historical vs recent modality mix ===")
print(
    era[
        [
            "modality_group_final",
            "2015–2022",
            "2023–2026 YTD",
            "historical_share",
            "recent_share",
            "share_change_pp",
        ]
    ].to_string(
        index=False,
        formatters={
            "historical_share": lambda x: f"{x:.1%}",
            "recent_share": lambda x: f"{x:.1%}",
            "share_change_pp": lambda x: f"{x:+.1f} pp",
        },
    )
)

print()
print("Saved tables:")
print(mapping_out)
print(overall_out)
print(trend_out)
print(era_out)
print(stage_out)

print()
print("Saved figures:")
print(fig_overall)
print(fig_era)
print(fig_trend)

print()
print("PHASE 5.5 MODALITY EVOLUTION COMPLETE.")
