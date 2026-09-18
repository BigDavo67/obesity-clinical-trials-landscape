from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed = Path("data/processed")
figures = Path("outputs/figures")
figures.mkdir(parents=True, exist_ok=True)

pair_file = processed / "phase5_trial_therapy_master.csv"

summary_out = processed / "phase56_mechanism_maturity_summary.csv"
identity_out = processed / "phase56_identity_highest_stage.csv"
stage_out = processed / "phase56_mechanism_identity_stage_matrix.csv"
recent_out = processed / "phase56_mechanism_recent_activity.csv"

fig_stage = figures / "phase56_mechanism_identity_stage_mix.png"
fig_scatter = figures / "phase56_mechanism_maturity_scatter.png"
fig_recent = figures / "phase56_recent_activity_vs_late_stage.png"

pair = pd.read_csv(pair_file)

print("=== Phase 5.6: pipeline maturity ===")
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
# 1. DEVELOPMENT-STAGE DEFINITIONS
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

verified["phase_rank"] = (
    verified["phase_group"]
    .map(phase_rank)
    .fillna(0)
    .astype(int)
)

verified["pipeline_stage"] = (
    verified["phase_group"].map(broad_stage)
)

verified["is_active_trial"] = (
    verified["status_group"].eq("Active / recruiting")
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
# 2. IDENTITY-LEVEL HIGHEST OBSERVED DEVELOPMENT STAGE
#    This avoids calling a mechanism "mature" just because one
#    asset has many trials.
# ------------------------------------------------------------

identity_rows = []

for (identity, mechanism), group in verified.groupby(
    ["therapy_identity_final", "mechanism_superfamily_final"]
):
    max_rank = int(group["phase_rank"].max())

    # Pick a representative highest phase deterministically.
    highest_phase_candidates = (
        group.loc[
            group["phase_rank"].eq(max_rank),
            "phase_group"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    highest_phase = (
        sorted(highest_phase_candidates)[0]
        if highest_phase_candidates
        else "Not applicable / not reported"
    )

    identity_rows.append({
        "therapy_identity_final": identity,
        "mechanism_superfamily_final": mechanism,
        "highest_phase_group": highest_phase,
        "highest_phase_rank": max_rank,
        "highest_stage_bucket": broad_stage(highest_phase),
        "trial_count": group["nct_id"].nunique(),
        "active_trial_count": group.loc[
            group["is_active_trial"], "nct_id"
        ].nunique(),
        "recent_confirmed_trial_count": group.loc[
            group["is_recent_confirmed"], "nct_id"
        ].nunique(),
        "has_active_trial": bool(group["is_active_trial"].any()),
        "has_recent_confirmed_trial": bool(
            group["is_recent_confirmed"].any()
        ),
        "first_start_year": int(group["start_year"].min()),
        "latest_start_year": int(group["start_year"].max()),
    })

identity = pd.DataFrame(identity_rows)

# Validate one mechanism per verified identity.
if identity["therapy_identity_final"].duplicated().any():
    bad = identity.loc[
        identity["therapy_identity_final"].duplicated(
            keep=False
        ),
        [
            "therapy_identity_final",
            "mechanism_superfamily_final",
        ]
    ]
    raise ValueError(
        "A verified therapy identity appears under multiple mechanisms:\n"
        + bad.to_string(index=False)
    )

identity.to_csv(identity_out, index=False)

# ------------------------------------------------------------
# 3. IDENTITY-STAGE MATRIX
# ------------------------------------------------------------

stage_order = [
    "Early development",
    "Mid development",
    "Late development",
    "Post-marketing",
    "Not applicable / not reported",
]

identity_stage = (
    identity.groupby(
        ["mechanism_superfamily_final", "highest_stage_bucket"]
    )
    .size()
    .unstack(fill_value=0)
)

for col in stage_order:
    if col not in identity_stage.columns:
        identity_stage[col] = 0

identity_stage = identity_stage[stage_order]
identity_stage["unique_therapy_identities"] = (
    identity_stage.sum(axis=1)
)

identity_stage = identity_stage.sort_values(
    "unique_therapy_identities",
    ascending=False,
)

identity_stage.to_csv(stage_out)

# ------------------------------------------------------------
# 4. PAIR-LEVEL ACTIVITY SUMMARY
# ------------------------------------------------------------

pair_stage = (
    verified.groupby(
        ["mechanism_superfamily_final", "pipeline_stage"]
    )
    .size()
    .unstack(fill_value=0)
)

for col in stage_order:
    if col not in pair_stage.columns:
        pair_stage[col] = 0

pair_stage = pair_stage[stage_order]

pair_stage["verified_pair_count"] = (
    pair_stage.sum(axis=1)
)

pair_stage["early_pair_share"] = (
    pair_stage["Early development"]
    / pair_stage["verified_pair_count"]
)

pair_stage["mid_pair_share"] = (
    pair_stage["Mid development"]
    / pair_stage["verified_pair_count"]
)

pair_stage["late_pair_share"] = (
    pair_stage["Late development"]
    / pair_stage["verified_pair_count"]
)

pair_stage["postmarketing_pair_share"] = (
    pair_stage["Post-marketing"]
    / pair_stage["verified_pair_count"]
)

# ------------------------------------------------------------
# 5. MECHANISM-LEVEL MATURITY SUMMARY
# ------------------------------------------------------------

mechanism_identity = (
    identity.groupby("mechanism_superfamily_final")
    .agg(
        unique_therapy_identities=(
            "therapy_identity_final",
            "nunique",
        ),
        active_therapy_identities=(
            "has_active_trial",
            "sum",
        ),
        recent_therapy_identities=(
            "has_recent_confirmed_trial",
            "sum",
        ),
        median_trials_per_identity=(
            "trial_count",
            "median",
        ),
    )
)

mechanism_identity["early_stage_identities"] = (
    identity[
        identity["highest_stage_bucket"]
        .eq("Early development")
    ]
    .groupby("mechanism_superfamily_final")
    .size()
)

mechanism_identity["mid_stage_identities"] = (
    identity[
        identity["highest_stage_bucket"]
        .eq("Mid development")
    ]
    .groupby("mechanism_superfamily_final")
    .size()
)

mechanism_identity["late_stage_identities"] = (
    identity[
        identity["highest_stage_bucket"]
        .eq("Late development")
    ]
    .groupby("mechanism_superfamily_final")
    .size()
)

mechanism_identity["postmarketing_identities"] = (
    identity[
        identity["highest_stage_bucket"]
        .eq("Post-marketing")
    ]
    .groupby("mechanism_superfamily_final")
    .size()
)

mechanism_identity = mechanism_identity.fillna(0)

for col in [
    "early_stage_identities",
    "mid_stage_identities",
    "late_stage_identities",
    "postmarketing_identities",
]:
    mechanism_identity[col] = (
        mechanism_identity[col].astype(int)
    )

mechanism_identity["phase3plus_identities"] = (
    mechanism_identity["late_stage_identities"]
    + mechanism_identity["postmarketing_identities"]
)

mechanism_identity["phase3plus_identity_share"] = (
    mechanism_identity["phase3plus_identities"]
    / mechanism_identity["unique_therapy_identities"]
)

mechanism_identity["early_identity_share"] = (
    mechanism_identity["early_stage_identities"]
    / mechanism_identity["unique_therapy_identities"]
)

mechanism_identity["recent_identity_share"] = (
    mechanism_identity["recent_therapy_identities"]
    / mechanism_identity["unique_therapy_identities"]
)

activity = (
    verified.groupby("mechanism_superfamily_final")
    .agg(
        verified_pair_count=("nct_id", "size"),
        unique_trials=("nct_id", "nunique"),
        active_verified_pairs=("is_active_trial", "sum"),
        recent_verified_pairs=("is_recent_confirmed", "sum"),
    )
)

activity["active_pair_share"] = (
    activity["active_verified_pairs"]
    / activity["verified_pair_count"]
)

activity["recent_pair_share"] = (
    activity["recent_verified_pairs"]
    / activity["verified_pair_count"]
)

summary = (
    mechanism_identity
    .join(activity, how="outer")
    .join(
        pair_stage[
            [
                "early_pair_share",
                "mid_pair_share",
                "late_pair_share",
                "postmarketing_pair_share",
            ]
        ],
        how="left",
    )
    .reset_index()
)

summary = summary.sort_values(
    [
        "unique_therapy_identities",
        "verified_pair_count",
    ],
    ascending=[False, False],
)

summary.to_csv(summary_out, index=False)

# ------------------------------------------------------------
# 6. RECENT-ACTIVITY TABLE
# ------------------------------------------------------------

recent = summary[
    [
        "mechanism_superfamily_final",
        "unique_therapy_identities",
        "recent_therapy_identities",
        "recent_identity_share",
        "recent_verified_pairs",
        "recent_pair_share",
        "phase3plus_identities",
        "phase3plus_identity_share",
        "early_stage_identities",
        "early_identity_share",
        "active_therapy_identities",
    ]
].copy()

recent = recent.sort_values(
    [
        "recent_therapy_identities",
        "recent_verified_pairs",
    ],
    ascending=[False, False],
)

recent.to_csv(recent_out, index=False)

# ------------------------------------------------------------
# 7. FIGURE — IDENTITY HIGHEST-STAGE MIX
# ------------------------------------------------------------

top15_mechs = (
    summary.head(15)["mechanism_superfamily_final"]
    .tolist()
)

stage_plot = (
    identity_stage
    .loc[
        [m for m in top15_mechs if m in identity_stage.index],
        stage_order,
    ]
    .copy()
)

plt.figure(figsize=(12, 8))

bottom = np.zeros(len(stage_plot))

for col in stage_order:
    values = stage_plot[col].values
    plt.barh(
        stage_plot.index,
        values,
        left=bottom,
        label=col,
    )
    bottom = bottom + values

plt.xlabel("Unique therapy identities")
plt.ylabel("Mechanism superfamily")
plt.title("Highest observed development stage by mechanism")
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
plt.tight_layout()
plt.savefig(
    fig_stage,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 8. FIGURE — MATURITY SCATTER
# ------------------------------------------------------------

scatter_df = summary[
    summary["unique_therapy_identities"] >= 2
].copy()

plt.figure(figsize=(11, 7))

plt.scatter(
    scatter_df["unique_therapy_identities"],
    scatter_df["phase3plus_identity_share"] * 100,
    s=(scatter_df["recent_therapy_identities"] + 1) * 35,
)

for _, row in scatter_df.iterrows():
    if (
        row["unique_therapy_identities"] >= 4
        or row["phase3plus_identities"] >= 2
        or row["recent_therapy_identities"] >= 4
    ):
        plt.annotate(
            row["mechanism_superfamily_final"],
            (
                row["unique_therapy_identities"],
                row["phase3plus_identity_share"] * 100,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

plt.xlabel("Unique therapy identities")
plt.ylabel("Therapy identities reaching Phase 3/4 (%)")
plt.title("Mechanism breadth vs development maturity")
plt.tight_layout()
plt.savefig(
    fig_scatter,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 9. FIGURE — RECENT ACTIVITY VS LATE-STAGE DEPTH
# ------------------------------------------------------------

recent_scatter = summary[
    summary["unique_therapy_identities"] >= 2
].copy()

plt.figure(figsize=(11, 7))

plt.scatter(
    recent_scatter["recent_identity_share"] * 100,
    recent_scatter["phase3plus_identity_share"] * 100,
    s=(recent_scatter["unique_therapy_identities"] + 1) * 30,
)

for _, row in recent_scatter.iterrows():
    if (
        row["unique_therapy_identities"] >= 4
        or row["recent_identity_share"] >= 0.75
        or row["phase3plus_identity_share"] >= 0.40
    ):
        plt.annotate(
            row["mechanism_superfamily_final"],
            (
                row["recent_identity_share"] * 100,
                row["phase3plus_identity_share"] * 100,
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

plt.xlabel("Therapy identities with recent confirmed activity (%)")
plt.ylabel("Therapy identities reaching Phase 3/4 (%)")
plt.title("Recent activity vs late-stage depth by mechanism")
plt.tight_layout()
plt.savefig(
    fig_recent,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 10. PRINT SUMMARY
# ------------------------------------------------------------

print()
print("Verified therapy identities:", len(identity))

print()
print("=== Top 15 mechanisms by unique therapy identities ===")
print(
    summary[
        [
            "mechanism_superfamily_final",
            "unique_therapy_identities",
            "verified_pair_count",
            "recent_therapy_identities",
            "active_therapy_identities",
            "early_stage_identities",
            "mid_stage_identities",
            "phase3plus_identities",
            "phase3plus_identity_share",
        ]
    ]
    .head(15)
    .to_string(
        index=False,
        formatters={
            "phase3plus_identity_share": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("=== Mechanisms with the most Phase 3/4 therapy identities ===")
late = summary.sort_values(
    [
        "phase3plus_identities",
        "unique_therapy_identities",
    ],
    ascending=[False, False],
).head(15)

print(
    late[
        [
            "mechanism_superfamily_final",
            "phase3plus_identities",
            "unique_therapy_identities",
            "phase3plus_identity_share",
            "recent_therapy_identities",
        ]
    ].to_string(
        index=False,
        formatters={
            "phase3plus_identity_share": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("=== Early-stage-heavy mechanisms (>=3 identities) ===")
early = summary[
    summary["unique_therapy_identities"] >= 3
].sort_values(
    [
        "early_identity_share",
        "recent_identity_share",
        "unique_therapy_identities",
    ],
    ascending=[False, False, False],
).head(15)

print(
    early[
        [
            "mechanism_superfamily_final",
            "unique_therapy_identities",
            "early_stage_identities",
            "early_identity_share",
            "recent_therapy_identities",
            "recent_identity_share",
            "phase3plus_identities",
        ]
    ].to_string(
        index=False,
        formatters={
            "early_identity_share": lambda x: f"{x:.1%}",
            "recent_identity_share": lambda x: f"{x:.1%}",
        },
    )
)

print()
print("Saved tables:")
print(summary_out)
print(identity_out)
print(stage_out)
print(recent_out)

print()
print("Saved figures:")
print(fig_stage)
print(fig_scatter)
print(fig_recent)

print()
print("PHASE 5.6 PIPELINE MATURITY COMPLETE.")
