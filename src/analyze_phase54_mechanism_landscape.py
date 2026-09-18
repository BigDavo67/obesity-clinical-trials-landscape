from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed = Path("data/processed")
figures = Path("outputs/figures")
figures.mkdir(parents=True, exist_ok=True)

pair_file = processed / "phase5_trial_therapy_master.csv"

overall_out = processed / "phase54_mechanism_landscape_overall.csv"
recent_out = processed / "phase54_mechanism_landscape_recent.csv"
coverage_out = processed / "phase54_mechanism_coverage_by_year.csv"
trend_out = processed / "phase54_mechanism_trend_by_year.csv"
stage_out = processed / "phase54_mechanism_by_pipeline_stage.csv"
identity_out = processed / "phase54_mechanism_identity_breadth.csv"

fig_overall = figures / "phase54_mechanism_landscape_overall.png"
fig_recent = figures / "phase54_mechanism_recent_vs_historical.png"
fig_stage = figures / "phase54_mechanism_stage_heatmap.png"
fig_coverage = figures / "phase54_mechanism_classification_coverage.png"

pair = pd.read_csv(pair_file)

print("=== Phase 5.4: mechanism landscape ===")
print("Input trial × focal-therapy pairs:", len(pair))

required = {
    "nct_id",
    "therapy_identity_final",
    "classification_status_final",
    "mechanism_superfamily_final",
    "start_year",
    "start_timing_vs_snapshot",
    "phase_group",
}
missing = required - set(pair.columns)
if missing:
    raise ValueError(
        f"Missing required columns from phase5_trial_therapy_master.csv: {sorted(missing)}"
    )

if len(pair) != 1186:
    raise ValueError(
        f"Expected 1186 trial × focal-therapy pairs; found {len(pair)}."
    )

verified_statuses = {"VERIFIED", "VERIFIED_ALIAS"}

pair["verified_mechanism"] = (
    pair["classification_status_final"]
    .isin(verified_statuses)
)

verified = pair[
    pair["verified_mechanism"]
].copy()

if len(verified) != 972:
    raise ValueError(
        f"Expected 972 verified mechanism pairs; found {len(verified)}."
    )

# ------------------------------------------------------------
# 1. DEFINE CONSERVATIVE TIME-TREND INCLUSION
# ------------------------------------------------------------

pair["include_in_confirmed_trend"] = (
    (pair["start_year"] < 2026)
    | (
        pair["start_year"].eq(2026)
        & pair["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT")
    )
)

verified["include_in_confirmed_trend"] = (
    (verified["start_year"] < 2026)
    | (
        verified["start_year"].eq(2026)
        & verified["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT")
    )
)

# ------------------------------------------------------------
# 2. OVERALL MECHANISM LANDSCAPE
# ------------------------------------------------------------

overall_pairs = (
    verified.groupby("mechanism_superfamily_final")
    .agg(
        verified_pair_count=("nct_id", "size"),
        unique_trials=("nct_id", "nunique"),
        unique_therapy_identities=("therapy_identity_final", "nunique"),
    )
    .reset_index()
)

overall_pairs["share_of_verified_pairs"] = (
    overall_pairs["verified_pair_count"] / len(verified)
)

overall_pairs = overall_pairs.sort_values(
    ["verified_pair_count", "unique_therapy_identities"],
    ascending=[False, False],
).reset_index(drop=True)

overall_pairs["rank_by_pair_count"] = np.arange(
    1, len(overall_pairs) + 1
)

overall_pairs.to_csv(overall_out, index=False)

# ------------------------------------------------------------
# 3. UNIQUE IDENTITY BREADTH BY MECHANISM
# ------------------------------------------------------------

verified_identity = (
    verified[
        [
            "therapy_identity_final",
            "mechanism_superfamily_final",
        ]
    ]
    .drop_duplicates()
)

# Conflicts should already be zero, but validate here too.
identity_conflicts = (
    verified_identity.groupby("therapy_identity_final")
    ["mechanism_superfamily_final"]
    .nunique()
)

if (identity_conflicts > 1).any():
    bad = identity_conflicts[
        identity_conflicts > 1
    ]
    raise ValueError(
        "A therapy identity maps to multiple mechanism superfamilies:\n"
        + bad.to_string()
    )

identity_breadth = (
    verified_identity.groupby("mechanism_superfamily_final")
    .agg(
        unique_therapy_identities=("therapy_identity_final", "nunique")
    )
    .reset_index()
)

identity_breadth["share_of_verified_identities"] = (
    identity_breadth["unique_therapy_identities"]
    / verified_identity["therapy_identity_final"].nunique()
)

identity_breadth = identity_breadth.sort_values(
    "unique_therapy_identities",
    ascending=False,
)

identity_breadth.to_csv(identity_out, index=False)

# ------------------------------------------------------------
# 4. CLASSIFICATION COVERAGE BY YEAR
# ------------------------------------------------------------

trend_pairs = pair[
    pair["include_in_confirmed_trend"]
].copy()

coverage = (
    trend_pairs.groupby("start_year")
    .agg(
        total_pairs=("nct_id", "size"),
        verified_pairs=("verified_mechanism", "sum"),
    )
    .reset_index()
)

coverage["unresolved_or_long_tail_pairs"] = (
    coverage["total_pairs"] - coverage["verified_pairs"]
)

coverage["verification_coverage"] = (
    coverage["verified_pairs"] / coverage["total_pairs"]
)

coverage.to_csv(coverage_out, index=False)

# ------------------------------------------------------------
# 5. MECHANISM TREND BY YEAR
# ------------------------------------------------------------

verified_trend = verified[
    verified["include_in_confirmed_trend"]
].copy()

trend = (
    verified_trend.groupby(
        ["start_year", "mechanism_superfamily_final"]
    )
    .size()
    .rename("verified_pair_count")
    .reset_index()
)

year_totals = (
    verified_trend.groupby("start_year")
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
# 6. HISTORICAL VS RECENT MECHANISM MIX
# ------------------------------------------------------------

verified_trend["era"] = np.where(
    verified_trend["start_year"].between(2015, 2022),
    "2015–2022",
    "2023–2026 YTD",
)

recent_table = (
    verified_trend.groupby(
        ["mechanism_superfamily_final", "era"]
    )
    .size()
    .unstack(fill_value=0)
)

for col in ["2015–2022", "2023–2026 YTD"]:
    if col not in recent_table.columns:
        recent_table[col] = 0

era_totals = (
    verified_trend.groupby("era")
    .size()
    .to_dict()
)

recent_table["historical_share"] = (
    recent_table["2015–2022"]
    / era_totals.get("2015–2022", np.nan)
)

recent_table["recent_share"] = (
    recent_table["2023–2026 YTD"]
    / era_totals.get("2023–2026 YTD", np.nan)
)

recent_table["share_change_pp"] = (
    (recent_table["recent_share"] - recent_table["historical_share"])
    * 100
)

recent_table = recent_table.reset_index().sort_values(
    ["2023–2026 YTD", "share_change_pp"],
    ascending=[False, False],
)

recent_table.to_csv(recent_out, index=False)

# ------------------------------------------------------------
# 7. PIPELINE-STAGE MIX BY MECHANISM
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
    verified["phase_group"]
    .map(broad_stage)
)

stage_order = [
    "Early development",
    "Mid development",
    "Late development",
    "Post-marketing",
    "Not applicable / not reported",
]

stage_matrix = (
    verified.groupby(
        ["mechanism_superfamily_final", "pipeline_stage"]
    )
    .size()
    .unstack(fill_value=0)
)

for col in stage_order:
    if col not in stage_matrix.columns:
        stage_matrix[col] = 0

stage_matrix = stage_matrix[stage_order]
stage_matrix["verified_pair_total"] = (
    stage_matrix.sum(axis=1)
)

stage_matrix = stage_matrix.sort_values(
    "verified_pair_total",
    ascending=False,
)

stage_matrix.to_csv(stage_out)

# ------------------------------------------------------------
# 8. FIGURE — OVERALL MECHANISM LANDSCAPE
# ------------------------------------------------------------

top15 = overall_pairs.head(15).sort_values(
    "verified_pair_count"
)

plt.figure(figsize=(10, 7))
plt.barh(
    top15["mechanism_superfamily_final"],
    top15["verified_pair_count"],
)
plt.title("Obesity pipeline by mechanism superfamily")
plt.xlabel("Verified trial × therapy pairs")
plt.ylabel("Mechanism superfamily")
plt.tight_layout()
plt.savefig(
    fig_overall,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 9. FIGURE — HISTORICAL VS RECENT SHARE
# ------------------------------------------------------------

recent_plot = (
    recent_table
    .sort_values("2023–2026 YTD", ascending=False)
    .head(12)
    .copy()
)

x = np.arange(len(recent_plot))
width = 0.38

plt.figure(figsize=(12, 7))
plt.bar(
    x - width / 2,
    recent_plot["historical_share"] * 100,
    width,
    label="2015–2022",
)
plt.bar(
    x + width / 2,
    recent_plot["recent_share"] * 100,
    width,
    label="2023–2026 YTD",
)
plt.xticks(
    x,
    recent_plot["mechanism_superfamily_final"],
    rotation=55,
    ha="right",
)
plt.ylabel("Share of verified mechanism pairs (%)")
plt.xlabel("Mechanism superfamily")
plt.title("How the obesity mechanism mix has shifted")
plt.legend()
plt.tight_layout()
plt.savefig(
    fig_recent,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 10. FIGURE — PIPELINE STAGE HEATMAP
# ------------------------------------------------------------

heat = (
    stage_matrix.head(15)[stage_order]
    .copy()
)

# Convert row counts to within-mechanism stage shares.
heat_share = heat.div(
    heat.sum(axis=1).replace(0, np.nan),
    axis=0,
)

plt.figure(figsize=(10, 8))
img = plt.imshow(
    heat_share.values,
    aspect="auto",
)
plt.colorbar(
    img,
    label="Share of verified pairs within mechanism",
)
plt.yticks(
    np.arange(len(heat_share)),
    heat_share.index,
)
plt.xticks(
    np.arange(len(stage_order)),
    stage_order,
    rotation=45,
    ha="right",
)

for i in range(heat_share.shape[0]):
    for j in range(heat_share.shape[1]):
        value = heat_share.iloc[i, j]
        if pd.notna(value):
            plt.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=8,
            )

plt.title("Pipeline maturity by mechanism superfamily")
plt.tight_layout()
plt.savefig(
    fig_stage,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 11. FIGURE — COVERAGE BY YEAR
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))
plt.plot(
    coverage["start_year"].astype(str),
    coverage["verification_coverage"] * 100,
    marker="o",
)
plt.ylim(0, 100)
plt.xlabel("Trial start year")
plt.ylabel("Mechanism verification coverage (%)")
plt.title("Mechanism-classification coverage by trial start year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(
    fig_coverage,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 12. PRINT SUMMARY
# ------------------------------------------------------------

print()
print("Verified mechanism pairs:", len(verified))
print(
    "Verified unique therapy identities:",
    verified_identity["therapy_identity_final"].nunique()
)

print()
print("=== Top 15 mechanism superfamilies overall ===")
print(
    overall_pairs[
        [
            "rank_by_pair_count",
            "mechanism_superfamily_final",
            "verified_pair_count",
            "share_of_verified_pairs",
            "unique_therapy_identities",
        ]
    ]
    .head(15)
    .to_string(
        index=False,
        formatters={
            "share_of_verified_pairs": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("=== Mechanism verification coverage by year ===")
print(
    coverage.to_string(
        index=False,
        formatters={
            "verification_coverage": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("=== Largest recent mechanism-share increases ===")
increase = recent_table.sort_values(
    "share_change_pp",
    ascending=False,
).head(12)

print(
    increase[
        [
            "mechanism_superfamily_final",
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
print("=== Largest recent mechanism-share decreases ===")
decrease = recent_table.sort_values(
    "share_change_pp",
    ascending=True,
).head(12)

print(
    decrease[
        [
            "mechanism_superfamily_final",
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
print(overall_out)
print(recent_out)
print(coverage_out)
print(trend_out)
print(stage_out)
print(identity_out)

print()
print("Saved figures:")
print(fig_overall)
print(fig_recent)
print(fig_stage)
print(fig_coverage)

print()
print("PHASE 5.4 MECHANISM LANDSCAPE COMPLETE.")
