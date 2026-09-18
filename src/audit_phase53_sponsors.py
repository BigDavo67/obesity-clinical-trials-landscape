from pathlib import Path
import pandas as pd
import re
import unicodedata

processed = Path("data/processed")

trial_file = processed / "phase5_trial_master.csv"
pair_file = processed / "phase5_trial_therapy_master.csv"

alias_out = processed / "phase53_sponsor_alias_audit.csv"
top_out = processed / "phase53_top_sponsors_raw.csv"

trial = pd.read_csv(trial_file)
pair = pd.read_csv(pair_file)

print("=== Phase 5.3A: sponsor-name audit ===")
print("Trial rows:", len(trial))
print("Pair rows:", len(pair))

required_trial = {
    "nct_id",
    "lead_sponsor_clean",
    "sponsor_class_clean",
    "start_year",
    "start_timing_vs_snapshot",
}
missing = required_trial - set(trial.columns)
if missing:
    raise ValueError(
        f"Missing required trial columns: {sorted(missing)}"
    )

# ------------------------------------------------------------
# 1. NORMALISED AUDIT KEY
#    This is ONLY for identifying likely duplicate labels.
#    It does not overwrite sponsor names.
# ------------------------------------------------------------

def sponsor_key(value):
    if pd.isna(value):
        return "unknown"

    s = str(value).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(
        ch for ch in s
        if not unicodedata.combining(ch)
    )

    replacements = {
        "&": " and ",
        "pharmaceuticals": "pharma",
        "pharmaceutical": "pharma",
        "incorporated": "inc",
        "corporation": "corp",
        "company": "co",
        "limited": "ltd",
    }

    for old, new in replacements.items():
        s = s.replace(old, new)

    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Remove common legal suffixes only at the end.
    suffixes = {
        "inc",
        "ltd",
        "llc",
        "plc",
        "corp",
        "co",
        "ag",
        "sa",
        "gmbh",
        "aps",
        "bv",
        "nv",
    }

    tokens = s.split()
    while tokens and tokens[-1] in suffixes:
        tokens.pop()

    return " ".join(tokens) if tokens else "unknown"

trial["sponsor_audit_key"] = (
    trial["lead_sponsor_clean"]
    .map(sponsor_key)
)

# ------------------------------------------------------------
# 2. TOP RAW SPONSORS
# ------------------------------------------------------------

top = (
    trial.groupby(
        ["lead_sponsor_clean", "sponsor_class_clean"],
        dropna=False,
    )
    .agg(
        trial_count=("nct_id", "nunique"),
        first_start_year=("start_year", "min"),
        latest_start_year=("start_year", "max"),
    )
    .reset_index()
    .sort_values(
        ["trial_count", "lead_sponsor_clean"],
        ascending=[False, True],
    )
)

top.to_csv(top_out, index=False)

# ------------------------------------------------------------
# 3. POTENTIAL ALIAS GROUPS
# ------------------------------------------------------------

alias_groups = (
    trial.groupby("sponsor_audit_key")
    .agg(
        unique_labels=("lead_sponsor_clean", "nunique"),
        labels=(
            "lead_sponsor_clean",
            lambda x: " | ".join(
                sorted(set(x.astype(str)))
            ),
        ),
        total_trials=("nct_id", "nunique"),
    )
    .reset_index()
)

alias_groups = alias_groups[
    alias_groups["unique_labels"] > 1
].sort_values(
    ["total_trials", "unique_labels"],
    ascending=[False, False],
)

alias_groups.to_csv(alias_out, index=False)

# ------------------------------------------------------------
# 4. RECENT-ACTIVITY VIEW
# ------------------------------------------------------------

recent = trial[
    trial["start_year"].between(2023, 2026)
    & (
        (trial["start_year"] < 2026)
        | trial["start_timing_vs_snapshot"].eq(
            "BEFORE_SNAPSHOT"
        )
    )
].copy()

recent_top = (
    recent.groupby("lead_sponsor_clean")
    .agg(
        recent_trial_count=("nct_id", "nunique")
    )
    .reset_index()
    .sort_values(
        ["recent_trial_count", "lead_sponsor_clean"],
        ascending=[False, True],
    )
)

# ------------------------------------------------------------
# 5. PRINT USEFUL AUDIT OUTPUT
# ------------------------------------------------------------

print()
print("Unique raw lead-sponsor labels:")
print(trial["lead_sponsor_clean"].nunique())

print()
print("Sponsor classes:")
print(
    trial["sponsor_class_clean"]
    .value_counts(dropna=False)
)

print()
print("=== Top 30 raw sponsors by unique trials ===")
print(
    top.head(30).to_string(index=False)
)

print()
print("=== Top 20 sponsors by confirmed 2023–2026 YTD starts ===")
print(
    recent_top.head(20).to_string(index=False)
)

print()
print("=== Potential duplicate/alias groups ===")
if alias_groups.empty:
    print("None found by deterministic audit key.")
else:
    print(
        alias_groups.head(30).to_string(index=False)
    )

print()
print("Saved:")
print(top_out)
print(alias_out)

print()
print("PHASE 5.3A SPONSOR AUDIT COMPLETE.")
