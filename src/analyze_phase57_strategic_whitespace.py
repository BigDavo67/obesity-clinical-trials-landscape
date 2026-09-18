from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed = Path("data/processed")
figures = Path("outputs/figures")
figures.mkdir(parents=True, exist_ok=True)

pair_file = processed / "phase5_trial_therapy_master.csv"

matrix_out = processed / "phase57_strategic_opportunity_matrix.csv"
emerging_out = processed / "phase57_emerging_early_stage_mechanisms.csv"
established_out = processed / "phase57_established_competitive_mechanisms.csv"
sponsor_out = processed / "phase57_mechanism_sponsor_density.csv"

fig_matrix = figures / "phase57_strategic_opportunity_matrix.png"
fig_sponsor = figures / "phase57_mechanism_sponsor_density.png"

pair = pd.read_csv(pair_file)

print("=== Phase 5.7: strategic whitespace / opportunity signals ===")
print("Input trial × focal-therapy pairs:", len(pair))

required = {
    "nct_id",
    "therapy_identity_final",
    "classification_status_final",
    "mechanism_superfamily_final",
    "phase_group",
    "status_group",
    "start_year",
    "start_timing_vs_snapshot",
    "lead_sponsor_clean",
    "sponsor_class_clean",
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

if len(verified) != 972:
    raise ValueError(
        f"Expected 972 verified pairs; found {len(verified)}."
    )

# ------------------------------------------------------------
# 1. CONSERVATIVE SPONSOR NORMALISATION
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

verified["sponsor_group_final"] = (
    verified["lead_sponsor_clean"]
    .map(normalization)
    .fillna(verified["lead_sponsor_clean"])
)

verified["is_industry"] = (
    verified["sponsor_class_clean"].eq("INDUSTRY")
)

verified["is_recent_confirmed"] = (
    verified["start_year"].between(2023, 2025)
    | (
        verified["start_year"].eq(2026)
        & verified["start_timing_vs_snapshot"].eq(
            "BEFORE_SNAPSHOT"
        )
    )
)

# ------------------------------------------------------------
# 2. IDENTITY HIGHEST-STAGE VIEW
# ------------------------------------------------------------

phase_rank = {
    "Not applicable / not reported": 0,
    "Early Phase 1": 1,
    "Phase 1": 2,
    "Phase 1/2": 3,
    "Phase 2": 4,
    "Phase 2/3": 5,
    "Phase 3": 6,
    "Phase 4": 7,
}

def broad_stage(phase):
    if phase in {"Early Phase 1", "Phase 1", "Phase 1/2"}:
        return "Early development"
    if phase in {"Phase 2", "Phase 2/3"}:
        return "Mid development"
    if phase == "Phase 3":
        return "Late development"
    if phase == "Phase 4":
        return "Post-marketing"
    return "Not applicable / not reported"

verified["phase_rank"] = (
    verified["phase_group"]
    .map(phase_rank)
    .fillna(0)
    .astype(int)
)

identity_rows = []

for (identity, mech), g in verified.groupby(
    ["therapy_identity_final", "mechanism_superfamily_final"]
):
    max_rank = int(g["phase_rank"].max())

    phase_candidates = (
        g.loc[g["phase_rank"].eq(max_rank), "phase_group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    highest_phase = (
        sorted(phase_candidates)[0]
        if phase_candidates
        else "Not applicable / not reported"
    )

    identity_rows.append({
        "therapy_identity_final": identity,
        "mechanism_superfamily_final": mech,
        "highest_stage_bucket": broad_stage(highest_phase),
        "has_recent_confirmed_activity": bool(
            g["is_recent_confirmed"].any()
        ),
        "has_active_trial": bool(
            g["status_group"].eq("Active / recruiting").any()
        ),
    })

identity = pd.DataFrame(identity_rows)

# ------------------------------------------------------------
# 3. MECHANISM BREADTH / MOMENTUM / MATURITY
# ------------------------------------------------------------

mech = (
    identity.groupby("mechanism_superfamily_final")
    .agg(
        unique_therapy_identities=(
            "therapy_identity_final",
            "nunique",
        ),
        recent_therapy_identities=(
            "has_recent_confirmed_activity",
            "sum",
        ),
        active_therapy_identities=(
            "has_active_trial",
            "sum",
        ),
    )
)

stage_counts = (
    identity.groupby(
        ["mechanism_superfamily_final", "highest_stage_bucket"]
    )
    .size()
    .unstack(fill_value=0)
)

for col in [
    "Early development",
    "Mid development",
    "Late development",
    "Post-marketing",
    "Not applicable / not reported",
]:
    if col not in stage_counts.columns:
        stage_counts[col] = 0

stage_counts["phase3plus_identities"] = (
    stage_counts["Late development"]
    + stage_counts["Post-marketing"]
)

mech = mech.join(
    stage_counts[
        [
            "Early development",
            "Mid development",
            "Late development",
            "Post-marketing",
            "phase3plus_identities",
        ]
    ],
    how="left",
).fillna(0)

mech["recent_identity_share"] = (
    mech["recent_therapy_identities"]
    / mech["unique_therapy_identities"]
)

mech["active_identity_share"] = (
    mech["active_therapy_identities"]
    / mech["unique_therapy_identities"]
)

mech["phase3plus_identity_share"] = (
    mech["phase3plus_identities"]
    / mech["unique_therapy_identities"]
)

# ------------------------------------------------------------
# 4. COMPETITIVE DENSITY
# ------------------------------------------------------------

industry = verified[
    verified["is_industry"]
].copy()

sponsor_density = (
    industry.groupby("mechanism_superfamily_final")
    .agg(
        industry_sponsor_groups=(
            "sponsor_group_final",
            "nunique",
        ),
        industry_trials=(
            "nct_id",
            "nunique",
        ),
    )
)

recent_industry = industry[
    industry["is_recent_confirmed"]
]

recent_sponsors = (
    recent_industry.groupby("mechanism_superfamily_final")
    ["sponsor_group_final"]
    .nunique()
    .rename("recent_industry_sponsor_groups")
)

sponsor_density = sponsor_density.join(
    recent_sponsors,
    how="left",
).fillna(0)

sponsor_density["recent_industry_sponsor_groups"] = (
    sponsor_density["recent_industry_sponsor_groups"]
    .astype(int)
)

matrix = (
    mech.join(
        sponsor_density,
        how="left",
    )
    .fillna({
        "industry_sponsor_groups": 0,
        "industry_trials": 0,
        "recent_industry_sponsor_groups": 0,
    })
    .reset_index()
)

for col in [
    "industry_sponsor_groups",
    "industry_trials",
    "recent_industry_sponsor_groups",
]:
    matrix[col] = matrix[col].astype(int)

matrix["industry_sponsors_per_identity"] = (
    matrix["industry_sponsor_groups"]
    / matrix["unique_therapy_identities"]
)

matrix["recent_sponsors_per_identity"] = (
    matrix["recent_industry_sponsor_groups"]
    / matrix["unique_therapy_identities"]
)

# ------------------------------------------------------------
# 5. TRANSPARENT STRATEGIC FLAGS
# ------------------------------------------------------------

matrix["emerging_early_stage_flag"] = (
    (matrix["unique_therapy_identities"] >= 3)
    & (matrix["recent_identity_share"] >= 0.50)
    & (matrix["phase3plus_identity_share"] <= 0.25)
)

matrix["established_competitive_flag"] = (
    (matrix["unique_therapy_identities"] >= 5)
    & (matrix["phase3plus_identity_share"] >= 0.35)
)

matrix["recent_momentum_flag"] = (
    (matrix["unique_therapy_identities"] >= 3)
    & (matrix["recent_identity_share"] >= 0.75)
)

matrix["low_late_stage_depth_flag"] = (
    (matrix["unique_therapy_identities"] >= 3)
    & (matrix["phase3plus_identity_share"] <= 0.20)
)

# Among emerging mechanisms only, compare sponsor density using
# the median of the emerging set rather than an arbitrary universal cut-off.
emerging_mask = matrix["emerging_early_stage_flag"]

if emerging_mask.any():
    emerging_sponsor_density_median = (
        matrix.loc[
            emerging_mask,
            "industry_sponsors_per_identity",
        ]
        .median()
    )
else:
    emerging_sponsor_density_median = np.nan

matrix["lower_sponsor_density_within_emerging"] = False

if pd.notna(emerging_sponsor_density_median):
    matrix.loc[
        emerging_mask
        & (
            matrix["industry_sponsors_per_identity"]
            <= emerging_sponsor_density_median
        ),
        "lower_sponsor_density_within_emerging"
    ] = True

def strategic_bucket(row):
    if row["emerging_early_stage_flag"]:
        if row["lower_sponsor_density_within_emerging"]:
            return "Emerging / lower sponsor density"
        return "Emerging / higher sponsor density"

    if row["established_competitive_flag"]:
        return "Established / competitive"

    if row["unique_therapy_identities"] < 3:
        return "Niche / limited evidence"

    return "Intermediate / mixed"

matrix["strategic_bucket"] = matrix.apply(
    strategic_bucket,
    axis=1,
)

# Sort for interpretation, not as a score/ranking.
matrix = matrix.sort_values(
    [
        "emerging_early_stage_flag",
        "recent_therapy_identities",
        "unique_therapy_identities",
    ],
    ascending=[False, False, False],
)

matrix.to_csv(matrix_out, index=False)

emerging = matrix[
    matrix["emerging_early_stage_flag"]
].copy()

emerging = emerging.sort_values(
    [
        "lower_sponsor_density_within_emerging",
        "recent_therapy_identities",
        "unique_therapy_identities",
    ],
    ascending=[False, False, False],
)

emerging.to_csv(emerging_out, index=False)

established = matrix[
    matrix["established_competitive_flag"]
].copy()

established = established.sort_values(
    [
        "phase3plus_identities",
        "unique_therapy_identities",
    ],
    ascending=[False, False],
)

established.to_csv(established_out, index=False)

sponsor_density.reset_index().to_csv(
    sponsor_out,
    index=False,
)

# ------------------------------------------------------------
# 6. FIGURE — OPPORTUNITY MATRIX
# ------------------------------------------------------------

plot_df = matrix[
    matrix["unique_therapy_identities"] >= 3
].copy()

plt.figure(figsize=(11, 8))

plt.scatter(
    plot_df["unique_therapy_identities"],
    plot_df["phase3plus_identity_share"] * 100,
    s=(plot_df["recent_therapy_identities"] + 1) * 45,
)

for _, row in plot_df.iterrows():
    if (
        row["emerging_early_stage_flag"]
        or row["established_competitive_flag"]
        or row["unique_therapy_identities"] >= 8
    ):
        plt.annotate(
            row["mechanism_superfamily_final"],
            (
                row["unique_therapy_identities"],
                row["phase3plus_identity_share"] * 100,
            ),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )

plt.xlabel("Unique therapy identities")
plt.ylabel("Therapy identities reaching Phase 3/4 (%)")
plt.title("Strategic mechanism landscape: breadth vs maturity")
plt.tight_layout()
plt.savefig(
    fig_matrix,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 7. FIGURE — SPONSOR DENSITY
# ------------------------------------------------------------

sponsor_plot = matrix[
    matrix["unique_therapy_identities"] >= 3
].sort_values(
    "industry_sponsors_per_identity"
)

plt.figure(figsize=(10, 7))
plt.barh(
    sponsor_plot["mechanism_superfamily_final"],
    sponsor_plot["industry_sponsors_per_identity"],
)
plt.xlabel("Industry sponsor groups per therapy identity")
plt.ylabel("Mechanism superfamily")
plt.title("Competitive density by mechanism")
plt.tight_layout()
plt.savefig(
    fig_sponsor,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 8. PRINT SUMMARY
# ------------------------------------------------------------

print()
print("=== Emerging / early-stage mechanisms ===")
if emerging.empty:
    print("None met the transparent emerging criteria.")
else:
    print(
        emerging[
            [
                "mechanism_superfamily_final",
                "unique_therapy_identities",
                "recent_therapy_identities",
                "recent_identity_share",
                "phase3plus_identities",
                "phase3plus_identity_share",
                "industry_sponsor_groups",
                "recent_industry_sponsor_groups",
                "industry_sponsors_per_identity",
                "strategic_bucket",
            ]
        ].to_string(
            index=False,
            formatters={
                "recent_identity_share": lambda x: f"{x:.1%}",
                "phase3plus_identity_share": lambda x: f"{x:.1%}",
                "industry_sponsors_per_identity": lambda x: f"{x:.2f}",
            },
        )
    )

print()
print("=== Established / competitive mechanisms ===")
if established.empty:
    print("None met the established criteria.")
else:
    print(
        established[
            [
                "mechanism_superfamily_final",
                "unique_therapy_identities",
                "phase3plus_identities",
                "phase3plus_identity_share",
                "recent_therapy_identities",
                "industry_sponsor_groups",
                "recent_industry_sponsor_groups",
            ]
        ].to_string(
            index=False,
            formatters={
                "phase3plus_identity_share": lambda x: f"{x:.1%}",
            },
        )
    )

print()
print("=== Mechanisms with strongest recent momentum (>=3 identities) ===")
momentum = matrix[
    matrix["unique_therapy_identities"] >= 3
].sort_values(
    [
        "recent_identity_share",
        "recent_therapy_identities",
        "unique_therapy_identities",
    ],
    ascending=[False, False, False],
).head(15)

print(
    momentum[
        [
            "mechanism_superfamily_final",
            "unique_therapy_identities",
            "recent_therapy_identities",
            "recent_identity_share",
            "phase3plus_identity_share",
            "industry_sponsor_groups",
            "strategic_bucket",
        ]
    ].to_string(
        index=False,
        formatters={
            "recent_identity_share": lambda x: f"{x:.1%}",
            "phase3plus_identity_share": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("Emerging-set median industry sponsors per identity:")
if pd.isna(emerging_sponsor_density_median):
    print("N/A")
else:
    print(f"{emerging_sponsor_density_median:.2f}")

print()
print("Saved tables:")
print(matrix_out)
print(emerging_out)
print(established_out)
print(sponsor_out)

print()
print("Saved figures:")
print(fig_matrix)
print(fig_sponsor)

print()
print("PHASE 5.7 STRATEGIC OPPORTUNITY ANALYSIS COMPLETE.")
