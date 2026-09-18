from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

processed = Path("data/processed")
figures = Path("outputs/figures")
figures.mkdir(parents=True, exist_ok=True)

trial_file = processed / "phase5_trial_master.csv"

yearly_out = processed / "phase52_trial_starts_by_year.csv"
phase_out = processed / "phase52_phase_mix_by_year.csv"
stage_out = processed / "phase52_pipeline_stage_by_year.csv"
status_out = processed / "phase52_status_by_year.csv"
metrics_out = processed / "phase52_pipeline_evolution_metrics.csv"

fig_yearly = figures / "phase52_trial_starts_over_time.png"
fig_phase = figures / "phase52_phase_mix_over_time.png"
fig_stage = figures / "phase52_pipeline_stage_over_time.png"

SNAPSHOT_DATE = "2026-09-17"

trial = pd.read_csv(trial_file)

print("=== Phase 5.2: pipeline evolution over time ===")
print("Input trials:", len(trial))

required = {
    "nct_id",
    "start_year",
    "start_timing_vs_snapshot",
    "phase_group",
    "status_group",
}
missing = required - set(trial.columns)
if missing:
    raise ValueError(
        f"Missing required columns from phase5_trial_master.csv: {sorted(missing)}"
    )

if len(trial) != 986:
    raise ValueError(
        f"Expected 986 trials; found {len(trial)}."
    )

if trial["start_year"].isna().any():
    raise ValueError(
        "start_year still contains missing values."
    )

trial["start_year"] = trial["start_year"].astype(int)

# ------------------------------------------------------------
# 1. DEFINE TREND INCLUSION RULES
# ------------------------------------------------------------

# Full historical years are complete within the frozen scope.
trial["include_in_confirmed_trend"] = (
    (trial["start_year"] < 2026)
    | (
        (trial["start_year"] == 2026)
        & trial["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT")
    )
)

# Upper-bound YTD adds September month-precision starts whose exact
# day relative to 17 Sep 2026 cannot be known from the registry.
trial["include_in_upper_bound_trend"] = (
    trial["include_in_confirmed_trend"]
    | (
        (trial["start_year"] == 2026)
        & trial["start_timing_vs_snapshot"].eq(
            "SNAPSHOT_MONTH_MONTH_PRECISION"
        )
    )
)

# Future planned 2026 starts are explicitly excluded from YTD.
future_2026 = trial[
    (trial["start_year"] == 2026)
    & trial["start_timing_vs_snapshot"].eq("AFTER_SNAPSHOT")
]

ambiguous_2026 = trial[
    (trial["start_year"] == 2026)
    & trial["start_timing_vs_snapshot"].eq(
        "SNAPSHOT_MONTH_MONTH_PRECISION"
    )
]

confirmed_2026 = trial[
    (trial["start_year"] == 2026)
    & trial["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT")
]

# ------------------------------------------------------------
# 2. YEARLY TRIAL STARTS
# ------------------------------------------------------------

years = list(range(2015, 2027))
rows = []

for year in years:
    year_df = trial[trial["start_year"] == year]

    if year < 2026:
        confirmed = len(year_df)
        ambiguous = 0
        future_planned = 0
        upper_bound = confirmed
        label = str(year)
    else:
        confirmed = len(confirmed_2026)
        ambiguous = len(ambiguous_2026)
        future_planned = len(future_2026)
        upper_bound = confirmed + ambiguous
        label = "2026 YTD"

    rows.append({
        "start_year": year,
        "year_label": label,
        "confirmed_started_count": confirmed,
        "ambiguous_same_month_count": ambiguous,
        "ytd_upper_bound_count": upper_bound,
        "future_planned_excluded": future_planned,
    })

yearly = pd.DataFrame(rows)

yearly["yoy_growth_confirmed"] = (
    yearly["confirmed_started_count"]
    .pct_change()
)

# 2026 YoY is not directly comparable because it is YTD.
yearly.loc[
    yearly["start_year"].eq(2026),
    "yoy_growth_confirmed"
] = np.nan

yearly.to_csv(yearly_out, index=False)

# ------------------------------------------------------------
# 3. PHASE MIX BY YEAR
# ------------------------------------------------------------

trend = trial[
    trial["include_in_confirmed_trend"]
].copy()

phase_order = [
    "Early Phase 1",
    "Phase 1",
    "Phase 1/2",
    "Phase 2",
    "Phase 2/3",
    "Phase 3",
    "Phase 4",
    "Not applicable / not reported",
]

phase_counts = (
    trend.groupby(
        ["start_year", "phase_group"]
    )
    .size()
    .unstack(fill_value=0)
    .reindex(
        index=years,
        columns=phase_order,
        fill_value=0,
    )
)

phase_counts.index.name = "start_year"
phase_counts.reset_index().to_csv(
    phase_out,
    index=False
)

# ------------------------------------------------------------
# 4. BROADER PIPELINE-STAGE VIEW
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

trend["pipeline_stage"] = trend[
    "phase_group"
].map(broad_stage)

stage_order = [
    "Early development",
    "Mid development",
    "Late development",
    "Post-marketing",
    "Not applicable / not reported",
]

stage_counts = (
    trend.groupby(
        ["start_year", "pipeline_stage"]
    )
    .size()
    .unstack(fill_value=0)
    .reindex(
        index=years,
        columns=stage_order,
        fill_value=0,
    )
)

stage_counts.index.name = "start_year"
stage_counts.reset_index().to_csv(
    stage_out,
    index=False
)

# ------------------------------------------------------------
# 5. CURRENT STATUS BY START COHORT
# ------------------------------------------------------------

status_order = [
    "Active / recruiting",
    "Completed",
    "Stopped early",
    "Unknown",
    "Other",
]

status_counts = (
    trend.groupby(
        ["start_year", "status_group"]
    )
    .size()
    .unstack(fill_value=0)
    .reindex(
        index=years,
        columns=status_order,
        fill_value=0,
    )
)

status_counts.index.name = "start_year"
status_counts.reset_index().to_csv(
    status_out,
    index=False
)

# ------------------------------------------------------------
# 6. SUMMARY METRICS
# ------------------------------------------------------------

starts_2015 = int(
    yearly.loc[
        yearly["start_year"].eq(2015),
        "confirmed_started_count"
    ].iloc[0]
)

starts_2023 = int(
    yearly.loc[
        yearly["start_year"].eq(2023),
        "confirmed_started_count"
    ].iloc[0]
)

starts_2024 = int(
    yearly.loc[
        yearly["start_year"].eq(2024),
        "confirmed_started_count"
    ].iloc[0]
)

starts_2025 = int(
    yearly.loc[
        yearly["start_year"].eq(2025),
        "confirmed_started_count"
    ].iloc[0]
)

starts_2026_confirmed = len(confirmed_2026)
starts_2026_upper = (
    len(confirmed_2026) + len(ambiguous_2026)
)

cagr_2015_2025 = (
    (starts_2025 / starts_2015) ** (1 / 10) - 1
)

growth_2023_2024 = (
    starts_2024 / starts_2023 - 1
)

growth_2024_2025 = (
    starts_2025 / starts_2024 - 1
)

recent_2023_2025 = trend[
    trend["start_year"].between(2023, 2025)
]

recent_early_share = (
    recent_2023_2025["phase_group"]
    .isin(
        ["Early Phase 1", "Phase 1", "Phase 1/2"]
    )
    .mean()
)

recent_late_share = (
    recent_2023_2025["phase_group"]
    .eq("Phase 3")
    .mean()
)

metrics = pd.DataFrame([
    {
        "metric": "trial_starts_2015",
        "value": starts_2015,
        "note": "Full-year trial starts",
    },
    {
        "metric": "trial_starts_2025",
        "value": starts_2025,
        "note": "Full-year trial starts",
    },
    {
        "metric": "2015_2025_cagr",
        "value": cagr_2015_2025,
        "note": "Compound annual growth in trial starts",
    },
    {
        "metric": "2023_2024_growth",
        "value": growth_2023_2024,
        "note": "Year-on-year growth in trial starts",
    },
    {
        "metric": "2024_2025_growth",
        "value": growth_2024_2025,
        "note": "Year-on-year growth in trial starts",
    },
    {
        "metric": "2026_ytd_confirmed_starts",
        "value": starts_2026_confirmed,
        "note": f"Definitely started by {SNAPSHOT_DATE}",
    },
    {
        "metric": "2026_ytd_ambiguous_september",
        "value": len(ambiguous_2026),
        "note": "Month-precision September starts; exact day unknown",
    },
    {
        "metric": "2026_ytd_upper_bound",
        "value": starts_2026_upper,
        "note": "Confirmed plus ambiguous September records",
    },
    {
        "metric": "2026_future_planned_excluded",
        "value": len(future_2026),
        "note": f"Reported start after {SNAPSHOT_DATE}",
    },
    {
        "metric": "2023_2025_early_development_share",
        "value": recent_early_share,
        "note": "Share of 2023-2025 starts in Early Phase 1 / Phase 1 / Phase 1/2",
    },
    {
        "metric": "2023_2025_phase3_share",
        "value": recent_late_share,
        "note": "Share of 2023-2025 starts in Phase 3",
    },
])

metrics.to_csv(metrics_out, index=False)

# ------------------------------------------------------------
# 7. FIGURE 1 — TRIAL STARTS OVER TIME
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

x = yearly["year_label"]
confirmed = yearly["confirmed_started_count"]
ambiguous = yearly["ambiguous_same_month_count"]

plt.bar(
    x,
    confirmed,
    label="Confirmed starts",
)

# Only visible for 2026 YTD.
plt.bar(
    x,
    ambiguous,
    bottom=confirmed,
    hatch="//",
    label="Sep 2026 month-only (timing ambiguous)",
)

plt.title("Obesity pharmacotherapy trial starts, 2015–2026 YTD")
plt.xlabel("Trial start year")
plt.ylabel("Number of trials")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(
    fig_yearly,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 8. FIGURE 2 — EXACT PHASE MIX
# ------------------------------------------------------------

phase_plot = phase_counts.copy()
phase_plot.index = [
    str(y) if y < 2026 else "2026 YTD"
    for y in phase_plot.index
]

plt.figure(figsize=(11, 7))

bottom = np.zeros(len(phase_plot))

for col in phase_order:
    values = phase_plot[col].values
    plt.bar(
        phase_plot.index,
        values,
        bottom=bottom,
        label=col,
    )
    bottom = bottom + values

plt.title("Phase mix of obesity pharmacotherapy trial starts")
plt.xlabel("Trial start year")
plt.ylabel("Number of trials")
plt.xticks(rotation=45)
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
)
plt.tight_layout()
plt.savefig(
    fig_phase,
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# ------------------------------------------------------------
# 9. FIGURE 3 — BROAD PIPELINE STAGE
# ------------------------------------------------------------

stage_plot = stage_counts.copy()
stage_plot.index = [
    str(y) if y < 2026 else "2026 YTD"
    for y in stage_plot.index
]

plt.figure(figsize=(11, 7))

bottom = np.zeros(len(stage_plot))

for col in stage_order:
    values = stage_plot[col].values
    plt.bar(
        stage_plot.index,
        values,
        bottom=bottom,
        label=col,
    )
    bottom = bottom + values

plt.title("Development-stage mix of obesity pharmacotherapy trial starts")
plt.xlabel("Trial start year")
plt.ylabel("Number of trials")
plt.xticks(rotation=45)
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
# 10. PRINT INTERPRETABLE OUTPUT
# ------------------------------------------------------------

print()
print("=== Annual trial starts ===")
print(
    yearly[
        [
            "year_label",
            "confirmed_started_count",
            "ambiguous_same_month_count",
            "future_planned_excluded",
        ]
    ].to_string(index=False)
)

print()
print("=== Key pipeline-evolution metrics ===")
print(
    f"2015 trial starts: {starts_2015}"
)
print(
    f"2025 trial starts: {starts_2025}"
)
print(
    f"2015–2025 CAGR: {cagr_2015_2025:.1%}"
)
print(
    f"2023→2024 growth: {growth_2023_2024:.1%}"
)
print(
    f"2024→2025 growth: {growth_2024_2025:.1%}"
)
print(
    f"2026 YTD confirmed starts: {starts_2026_confirmed}"
)
print(
    f"2026 YTD upper bound including 9 ambiguous September records: {starts_2026_upper}"
)
print(
    f"2026 future planned starts excluded: {len(future_2026)}"
)
print(
    f"2023–2025 early-development share: {recent_early_share:.1%}"
)
print(
    f"2023–2025 Phase 3 share: {recent_late_share:.1%}"
)

print()
print("Saved tables:")
print(yearly_out)
print(phase_out)
print(stage_out)
print(status_out)
print(metrics_out)

print()
print("Saved figures:")
print(fig_yearly)
print(fig_phase)
print(fig_stage)

print()
print("PHASE 5.2 PIPELINE EVOLUTION COMPLETE.")
