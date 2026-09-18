from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed = Path("data/processed")
figures = Path("outputs/figures")
figures.mkdir(parents=True, exist_ok=True)

trial_file = processed / "phase5_trial_master.csv"
pair_file = processed / "phase5_trial_therapy_master.csv"

sponsor_table_out = processed / "phase53_sponsor_competitive_landscape.csv"
concentration_out = processed / "phase53_market_concentration.csv"
mechanism_out = processed / "phase53_sponsor_mechanism_matrix.csv"
normalization_out = processed / "phase53_sponsor_normalization_map.csv"

fig_trials = figures / "phase53_top_industry_sponsors_by_trials.png"
fig_recent = figures / "phase53_top_industry_sponsors_recent_activity.png"
fig_breadth = figures / "phase53_sponsor_portfolio_breadth.png"

trial = pd.read_csv(trial_file)
pair = pd.read_csv(pair_file)

print("=== Phase 5.3: sponsor competitive landscape ===")
print("Trial rows:", len(trial))
print("Pair rows:", len(pair))

if len(trial) != 986:
    raise ValueError(f"Expected 986 trial rows; found {len(trial)}.")
if len(pair) != 1186:
    raise ValueError(f"Expected 1186 pair rows; found {len(pair)}.")

# ------------------------------------------------------------
# 1. CONSERVATIVE CORPORATE SPONSOR NORMALISATION
# ------------------------------------------------------------

normalization = {
    "Novo Nordisk A/S": "Novo Nordisk",
    "Eli Lilly and Company": "Eli Lilly",
    "Rhythm Pharmaceuticals, Inc.": "Rhythm Pharmaceuticals",
    "Gan & Lee Pharmaceuticals.": "Gan & Lee Pharmaceuticals",
    "Gan and Lee Pharmaceuticals, USA": "Gan & Lee Pharmaceuticals",
    "Gasherbrum Bio, Inc., a wholly owned subsidiary of Structure Therapeutics": "Structure Therapeutics",
    "Hoffmann-La Roche": "Roche",
}

normalization_rows = []
for raw_name in sorted(trial["lead_sponsor_clean"].dropna().unique()):
    group = normalization.get(raw_name, raw_name)
    rule = "MANUAL_CORPORATE_ALIAS" if raw_name in normalization else "UNCHANGED"
    normalization_rows.append({
        "lead_sponsor_raw": raw_name,
        "sponsor_group_final": group,
        "normalization_rule": rule,
    })

normalization_df = pd.DataFrame(normalization_rows)
normalization_df.to_csv(normalization_out, index=False)

trial["sponsor_group_final"] = (
    trial["lead_sponsor_clean"]
    .map(normalization)
    .fillna(trial["lead_sponsor_clean"])
)

# Pair-level sponsor assignment comes from trial master to prevent drift.
pair = pair.drop(columns=["sponsor_group_final"], errors="ignore")
pair = pair.merge(
    trial[["nct_id", "sponsor_group_final"]],
    on="nct_id",
    how="left",
    validate="many_to_one",
)

# ------------------------------------------------------------
# 2. ANALYSIS FLAGS
# ------------------------------------------------------------

trial["confirmed_recent_start"] = (
    trial["start_year"].between(2023, 2025)
    | (
        trial["start_year"].eq(2026)
        & trial["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT")
    )
)

trial["active_now"] = trial["status_group"].eq("Active / recruiting")

trial["phase3_or_later"] = trial["phase_group"].isin(
    ["Phase 3", "Phase 4"]
)

trial["early_stage"] = trial["phase_group"].isin(
    ["Early Phase 1", "Phase 1", "Phase 1/2"]
)

industry_trials = trial[
    trial["sponsor_class_clean"].eq("INDUSTRY")
].copy()

industry_ids = set(industry_trials["nct_id"])

industry_pairs = pair[
    pair["nct_id"].isin(industry_ids)
].copy()

# ------------------------------------------------------------
# 3. SPONSOR-LEVEL METRICS
# ------------------------------------------------------------

trial_metrics = (
    industry_trials.groupby("sponsor_group_final")
    .agg(
        total_trials=("nct_id", "nunique"),
        recent_confirmed_starts=("confirmed_recent_start", "sum"),
        active_trials=("active_now", "sum"),
        early_stage_trials=("early_stage", "sum"),
        phase3_or_later_trials=("phase3_or_later", "sum"),
        first_start_year=("start_year", "min"),
        latest_start_year=("start_year", "max"),
    )
    .reset_index()
)

verified_pairs = industry_pairs[
    industry_pairs["classification_status_final"].isin(
        ["VERIFIED", "VERIFIED_ALIAS"]
    )
].copy()

portfolio_metrics = (
    industry_pairs.groupby("sponsor_group_final")
    .agg(
        unique_therapy_identities=("therapy_identity_final", "nunique"),
        focal_pair_count=("nct_id", "size"),
    )
    .reset_index()
)

mechanism_metrics = (
    verified_pairs.groupby("sponsor_group_final")
    .agg(
        verified_pair_count=("nct_id", "size"),
        mechanism_superfamily_breadth=(
            "mechanism_superfamily_final",
            "nunique",
        ),
    )
    .reset_index()
)

sponsor = (
    trial_metrics
    .merge(portfolio_metrics, on="sponsor_group_final", how="left")
    .merge(mechanism_metrics, on="sponsor_group_final", how="left")
)

for col in [
    "unique_therapy_identities",
    "focal_pair_count",
    "verified_pair_count",
    "mechanism_superfamily_breadth",
]:
    sponsor[col] = sponsor[col].fillna(0).astype(int)

total_industry_trials = industry_trials["nct_id"].nunique()
recent_industry_trials = int(industry_trials["confirmed_recent_start"].sum())

sponsor["share_of_industry_trials"] = (
    sponsor["total_trials"] / total_industry_trials
)

sponsor["share_of_recent_industry_starts"] = (
    sponsor["recent_confirmed_starts"] / recent_industry_trials
    if recent_industry_trials
    else np.nan
)

sponsor["active_trial_share"] = (
    sponsor["active_trials"] / sponsor["total_trials"]
)

sponsor["phase3plus_share"] = (
    sponsor["phase3_or_later_trials"] / sponsor["total_trials"]
)

sponsor["trials_per_therapy_identity"] = (
    sponsor["total_trials"]
    / sponsor["unique_therapy_identities"].replace(0, np.nan)
)

sponsor = sponsor.sort_values(
    ["total_trials", "unique_therapy_identities", "sponsor_group_final"],
    ascending=[False, False, True],
).reset_index(drop=True)

sponsor["trial_volume_rank"] = np.arange(1, len(sponsor) + 1)

sponsor.to_csv(sponsor_table_out, index=False)

# ------------------------------------------------------------
# 4. MARKET CONCENTRATION
# ------------------------------------------------------------

shares = sponsor["share_of_industry_trials"]

hhi = float((shares ** 2).sum())
hhi_10000 = hhi * 10000

def top_share(n):
    return float(
        sponsor.head(n)["share_of_industry_trials"].sum()
    )

concentration = pd.DataFrame([
    {
        "metric": "industry_trials",
        "value": total_industry_trials,
    },
    {
        "metric": "industry_sponsor_groups",
        "value": sponsor["sponsor_group_final"].nunique(),
    },
    {
        "metric": "top_2_share",
        "value": top_share(2),
    },
    {
        "metric": "top_5_share",
        "value": top_share(5),
    },
    {
        "metric": "top_10_share",
        "value": top_share(10),
    },
    {
        "metric": "HHI_0_to_1",
        "value": hhi,
    },
    {
        "metric": "HHI_0_to_10000",
        "value": hhi_10000,
    },
    {
        "metric": "recent_industry_confirmed_starts_2023_2026YTD",
        "value": recent_industry_trials,
    },
])

concentration.to_csv(concentration_out, index=False)

# ------------------------------------------------------------
# 5. SPONSOR × MECHANISM MATRIX
# ------------------------------------------------------------

mechanism_matrix = (
    verified_pairs.groupby(
        ["sponsor_group_final", "mechanism_superfamily_final"]
    )
    .size()
    .unstack(fill_value=0)
)

mechanism_matrix["verified_pair_total"] = mechanism_matrix.sum(axis=1)

mechanism_matrix = mechanism_matrix.sort_values(
    "verified_pair_total",
    ascending=False,
)

mechanism_matrix.to_csv(mechanism_out)

# ------------------------------------------------------------
# 6. FIGURES
# ------------------------------------------------------------

top15 = sponsor.head(15).sort_values("total_trials")

plt.figure(figsize=(10, 7))
plt.barh(
    top15["sponsor_group_final"],
    top15["total_trials"],
)
plt.title("Leading industry sponsors by obesity pharmacotherapy trials")
plt.xlabel("Unique trials")
plt.ylabel("Lead sponsor group")
plt.tight_layout()
plt.savefig(fig_trials, dpi=200, bbox_inches="tight")
plt.close()

recent_top = (
    sponsor.sort_values(
        ["recent_confirmed_starts", "total_trials"],
        ascending=[False, False],
    )
    .head(15)
    .sort_values("recent_confirmed_starts")
)

plt.figure(figsize=(10, 7))
plt.barh(
    recent_top["sponsor_group_final"],
    recent_top["recent_confirmed_starts"],
)
plt.title("Leading industry sponsors by recent trial starts (2023–2026 YTD)")
plt.xlabel("Confirmed trial starts")
plt.ylabel("Lead sponsor group")
plt.tight_layout()
plt.savefig(fig_recent, dpi=200, bbox_inches="tight")
plt.close()

breadth_top = (
    sponsor[
        sponsor["unique_therapy_identities"] > 0
    ]
    .sort_values(
        ["unique_therapy_identities", "total_trials"],
        ascending=[False, False],
    )
    .head(15)
)

plt.figure(figsize=(10, 7))
plt.scatter(
    breadth_top["unique_therapy_identities"],
    breadth_top["total_trials"],
    s=(breadth_top["mechanism_superfamily_breadth"] + 1) * 35,
)

for _, row in breadth_top.iterrows():
    plt.annotate(
        row["sponsor_group_final"],
        (
            row["unique_therapy_identities"],
            row["total_trials"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

plt.title("Sponsor portfolio breadth vs clinical-trial intensity")
plt.xlabel("Unique focal therapy identities")
plt.ylabel("Unique trials")
plt.tight_layout()
plt.savefig(fig_breadth, dpi=200, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 7. PRINT SUMMARY
# ------------------------------------------------------------

print()
print("=== Industry competitive landscape ===")
print("Industry trials:", total_industry_trials)
print("Industry sponsor groups:", sponsor["sponsor_group_final"].nunique())
print("Recent confirmed industry starts (2023–2026 YTD):", recent_industry_trials)

print()
print("=== Concentration ===")
print(f"Top 2 share of industry trials: {top_share(2):.1%}")
print(f"Top 5 share of industry trials: {top_share(5):.1%}")
print(f"Top 10 share of industry trials: {top_share(10):.1%}")
print(f"HHI: {hhi_10000:.0f} on 0–10,000 scale")

cols = [
    "trial_volume_rank",
    "sponsor_group_final",
    "total_trials",
    "recent_confirmed_starts",
    "active_trials",
    "unique_therapy_identities",
    "mechanism_superfamily_breadth",
    "phase3_or_later_trials",
    "share_of_industry_trials",
]

print()
print("=== Top 20 industry sponsors ===")
print(
    sponsor[cols]
    .head(20)
    .to_string(
        index=False,
        formatters={
            "share_of_industry_trials": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("=== Top 15 by unique therapy identities ===")
breadth_print = (
    sponsor.sort_values(
        ["unique_therapy_identities", "total_trials"],
        ascending=[False, False],
    )
    .head(15)
)

print(
    breadth_print[
        [
            "sponsor_group_final",
            "unique_therapy_identities",
            "total_trials",
            "mechanism_superfamily_breadth",
            "recent_confirmed_starts",
        ]
    ].to_string(index=False)
)

print()
print("Saved:")
print(sponsor_table_out)
print(concentration_out)
print(mechanism_out)
print(normalization_out)

print()
print("Saved figures:")
print(fig_trials)
print(fig_recent)
print(fig_breadth)

print()
print("PHASE 5.3 SPONSOR COMPETITIVE LANDSCAPE COMPLETE.")
