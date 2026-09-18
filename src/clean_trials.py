from pathlib import Path
import json
import re
import unicodedata
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

processed_data_dir = Path("data/processed")

trials_file = processed_data_dir / "candidate_trials.csv"
interventions_file = processed_data_dir / "candidate_interventions.csv"
locations_file = processed_data_dir / "candidate_locations.csv"
collaborators_file = processed_data_dir / "candidate_collaborators.csv"

raw_file = Path("data/raw/obesity_trials_raw.json")


# ============================================================
# 2. LOAD CANDIDATE DATASETS
# ============================================================

trials_df = pd.read_csv(trials_file)
interventions_df = pd.read_csv(interventions_file)
locations_df = pd.read_csv(locations_file)
collaborators_df = pd.read_csv(collaborators_file)

print("Candidate datasets loaded")
print()
print("Trials:", len(trials_df))
print("Interventions:", len(interventions_df))
print("Locations:", len(locations_df))
print("Collaborators:", len(collaborators_df))


# ============================================================
# 3. LOAD AND VALIDATE FROZEN RAW JSON SNAPSHOT
# ============================================================

with open(raw_file, "r", encoding="utf-8") as file:
    raw_data = json.load(file)

raw_study_count = len(raw_data["studies"])

print()
print("Raw JSON studies:", raw_study_count)
print("Candidate CSV trials:", len(trials_df))

if raw_study_count != len(trials_df):
    raise ValueError(
        "Raw JSON and candidate CSV files come from different snapshots."
    )


# ============================================================
# 4. EXTRACT PRIMARY OUTCOMES FROM RAW JSON
# ============================================================

primary_outcome_rows = []

for study in raw_data["studies"]:

    protocol = study.get("protocolSection", {})

    identification = protocol.get("identificationModule", {})
    outcomes_module = protocol.get("outcomesModule", {})

    nct_id = identification.get("nctId")
    primary_outcomes = outcomes_module.get("primaryOutcomes", [])

    for outcome in primary_outcomes:

        row = {
            "nct_id": nct_id,
            "primary_outcome_measure": outcome.get("measure"),
            "primary_outcome_timeframe": outcome.get("timeFrame"),
            "primary_outcome_description": outcome.get("description")
        }

        primary_outcome_rows.append(row)

primary_outcomes_df = pd.DataFrame(primary_outcome_rows)

print()
print("=== Primary outcomes ===")
print("Rows:", len(primary_outcomes_df))

print(
    "Unique trials with primary outcomes:",
    primary_outcomes_df["nct_id"].nunique()
)

print(
    "Trials with no primary outcome rows:",
    len(trials_df) - primary_outcomes_df["nct_id"].nunique()
)

print()
print(primary_outcomes_df.head(10).to_string(index=False))


def join_unique_nonempty(values):
    cleaned_values = [
        str(value).strip()
        for value in values.dropna()
        if str(value).strip()
    ]

    return "; ".join(dict.fromkeys(cleaned_values))


primary_outcomes_by_trial = (
    primary_outcomes_df
    .groupby("nct_id")
    .agg(
        primary_outcome_measures=(
            "primary_outcome_measure",
            join_unique_nonempty
        ),
        primary_outcome_timeframes=(
            "primary_outcome_timeframe",
            join_unique_nonempty
        ),
        primary_outcome_descriptions=(
            "primary_outcome_description",
            join_unique_nonempty
        )
    )
    .reset_index()
)


# ============================================================
# 5. PHASE 4.2 — APPLY 2015–2026 START-YEAR SCOPE
# ============================================================

trials_df["start_date_parsed"] = pd.to_datetime(
    trials_df["start_date"],
    format="mixed",
    errors="coerce"
)

trials_df["start_year"] = (
    trials_df["start_date_parsed"]
    .dt.year
    .astype("Int64")
)

in_scope_mask = trials_df["start_year"].between(2015, 2026)
before_2015_mask = trials_df["start_year"] < 2015
after_2026_mask = trials_df["start_year"] > 2026
missing_start_year_mask = trials_df["start_year"].isna()

print()
print("=== Start-year scope ===")

print("Candidate trials:", len(trials_df))
print("Started 2015-2026:", in_scope_mask.sum())
print("Started before 2015:", before_2015_mask.sum())
print("Started after 2026:", after_2026_mask.sum())
print("Missing start year:", missing_start_year_mask.sum())

trials_scoped_df = trials_df[
    in_scope_mask
].copy()

scoped_trial_ids = set(trials_scoped_df["nct_id"])

interventions_scoped_df = interventions_df[
    interventions_df["nct_id"].isin(scoped_trial_ids)
].copy()

locations_scoped_df = locations_df[
    locations_df["nct_id"].isin(scoped_trial_ids)
].copy()

collaborators_scoped_df = collaborators_df[
    collaborators_df["nct_id"].isin(scoped_trial_ids)
].copy()

print()
print("Scoped dataset sizes:")
print("Trials:", len(trials_scoped_df))
print("Interventions:", len(interventions_scoped_df))
print("Locations:", len(locations_scoped_df))
print("Collaborators:", len(collaborators_scoped_df))


# ============================================================
# 6. PHASE 4.3 — REMOVE EXACT DUPLICATE RECORDS
# ============================================================

print()
print("=== Exact duplicate check ===")

print(
    "Duplicate trial rows:",
    trials_scoped_df.duplicated().sum()
)

print(
    "Duplicate intervention rows:",
    interventions_scoped_df.duplicated().sum()
)

print(
    "Duplicate location rows:",
    locations_scoped_df.duplicated().sum()
)

print(
    "Duplicate collaborator rows:",
    collaborators_scoped_df.duplicated().sum()
)

trials_clean_df = (
    trials_scoped_df
    .drop_duplicates()
    .copy()
)

interventions_clean_df = (
    interventions_scoped_df
    .drop_duplicates()
    .copy()
)

locations_clean_df = (
    locations_scoped_df
    .drop_duplicates()
    .copy()
)

collaborators_clean_df = (
    collaborators_scoped_df
    .drop_duplicates()
    .copy()
)

print()
print("Rows removed as exact duplicates:")

print(
    "Trials:",
    len(trials_scoped_df) - len(trials_clean_df)
)

print(
    "Interventions:",
    len(interventions_scoped_df) - len(interventions_clean_df)
)

print(
    "Locations:",
    len(locations_scoped_df) - len(locations_clean_df)
)

print(
    "Collaborators:",
    len(collaborators_scoped_df) - len(collaborators_clean_df)
)

print()
print("Dataset sizes after exact deduplication:")

print("Trials:", len(trials_clean_df))
print("Interventions:", len(interventions_clean_df))
print("Locations:", len(locations_clean_df))
print("Collaborators:", len(collaborators_clean_df))


# ============================================================
# 7. PHASE 4.4A — BUILD TRIAL REVIEW TABLE
# ============================================================

pharmacological_types = [
    "DRUG",
    "BIOLOGICAL",
    "COMBINATION_PRODUCT"
]

pharmacological_interventions_df = interventions_clean_df[
    interventions_clean_df["intervention_type"].isin(
        pharmacological_types
    )
].copy()

trial_interventions = (
    pharmacological_interventions_df
    .groupby("nct_id")["intervention_name"]
    .apply(
        lambda names: "; ".join(
            sorted(set(names.dropna()))
        )
    )
    .reset_index()
)

trial_interventions = trial_interventions.rename(
    columns={
        "intervention_name": "pharmacological_interventions"
    }
)

trial_review_df = trials_clean_df.merge(
    trial_interventions,
    on="nct_id",
    how="left"
)

trial_review_df = trial_review_df.merge(
    primary_outcomes_by_trial,
    on="nct_id",
    how="left"
)

print()
print("=== Trial review table ===")

print("Rows:", len(trial_review_df))

print(
    "Trials with no pharmacological intervention listed:",
    trial_review_df[
        "pharmacological_interventions"
    ].isna().sum()
)

review_columns = [
    "nct_id",
    "brief_title",
    "conditions",
    "pharmacological_interventions",
    "primary_purpose",
    "phase"
]

print()
print(
    trial_review_df[review_columns]
    .head(20)
    .to_string(index=False)
)

unique_pharmacological_interventions = (
    pharmacological_interventions_df[
        "intervention_name"
    ]
    .nunique()
)

print()
print(
    "Unique pharmacological intervention names:",
    unique_pharmacological_interventions
)


# ============================================================
# 8. PHASE 4.4C — CREATE AUTOMATED REVIEW SIGNALS
# ============================================================

text_columns = [
    "brief_title",
    "official_title",
    "conditions",
    "brief_summary",
    "pharmacological_interventions",
    "primary_outcome_measures",
    "primary_outcome_descriptions"
]

trial_review_df["review_text"] = (
    trial_review_df[text_columns]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

include_phrases = [
    "weight loss",
    "weight-loss",
    "weight reduction",
    "weight-reduction",
    "reduce body weight",
    "body weight reduction",
    "body-weight reduction",
    "weight management",
    "weight-management",
    "weight maintenance",
    "weight-maintenance",
    "maintain weight loss",
    "maintain weight-loss",
    "weight regain",
    "weight-regain",
    "prevent weight regain",
    "prevent weight-regain",
    "anti-obesity",
    "antiobesity",
    "anti-obesity medication",
    "anti-obesity medications",
    "obesity treatment",
    "treatment of obesity",
    "treat obesity",
    "treating obesity",
    "excess body weight",
    "hypothalamic obesity",
    "genetic obesity",
    "obesity disease"
]

exclude_phrases = [
    "bariatric surgery",
    "metabolic surgery",
    "postoperative",
    "post-operative",
    "perioperative",
    "peri-operative",
    "anesthesia",
    "anaesthesia",
    "analgesia",
    "postoperative pain",
    "post-operative pain",
    "preeclampsia",
    "pre-eclampsia",
    "surgical complications",
    "after surgery"
]


def contains_any(text, phrases):
    return any(
        phrase in text
        for phrase in phrases
    )


trial_review_df["include_signal"] = (
    trial_review_df["review_text"]
    .apply(
        lambda text: contains_any(
            text,
            include_phrases
        )
    )
)

trial_review_df["exclude_signal"] = (
    trial_review_df["review_text"]
    .apply(
        lambda text: contains_any(
            text,
            exclude_phrases
        )
    )
)


# ============================================================
# 9. PRIMARY WEIGHT-OUTCOME SIGNAL
# ============================================================

trial_review_df["primary_outcome_text"] = (
    trial_review_df[
        [
            "primary_outcome_measures",
            "primary_outcome_descriptions"
        ]
    ]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
    .str.lower()
)

weight_outcome_phrases = [
    "change in body weight",
    "change from baseline in body weight",
    "percent change in body weight",
    "percentage change in body weight",
    "body weight reduction",
    "weight reduction",
    "weight loss",
    "weight-loss",
    "total weight loss",
    "%twl",
    "percentage of body weight",
    "percent body weight",
    "bodyweight reduction"
]

trial_review_df["direct_weight_outcome_signal"] = (
    trial_review_df["primary_outcome_text"]
    .apply(
        lambda text: contains_any(
            text,
            weight_outcome_phrases
        )
    )
)

print()
print(
    "Direct weight-outcome trials:",
    trial_review_df[
        "direct_weight_outcome_signal"
    ].sum()
)


# ============================================================
# 10. PHASE 4.5/4.6 — CONTROL CLEANING + PROGRAMME EVIDENCE
# ============================================================

# The intervention table sometimes contains strings such as:
#   "HRS9531 injection；Placebo"
#   "Treatment with placebo; Treatment with tirzapatide"
#
# We do NOT want to discard the whole row just because the word
# placebo appears. Instead, split multi-part names, remove components
# that are clearly control-only, then normalise the remaining active
# intervention text for programme matching.

control_only_phrases = [
    "placebo",
    "matched placebo",
    "matching placebo",
    "normal saline",
    "saline",
    "vehicle",
    "sham"
]


def normalise_unicode_text(value):
    if pd.isna(value):
        return ""

    return unicodedata.normalize(
        "NFKC",
        str(value)
    ).strip()


def is_control_only_component(component):
    text = normalise_unicode_text(component).lower().strip()

    if not text:
        return True

    # Control descriptions commonly begin with one of these terms,
    # e.g. "Placebo", "Placebo matched to...", "Normal saline infusion".
    return any(
        text.startswith(phrase)
        for phrase in control_only_phrases
    )


def active_components_from_name(name):
    text = normalise_unicode_text(name)

    # NFKC converts many full-width characters, but explicitly handle
    # semicolon-like separators as well for robustness.
    text = text.replace("；", ";")

    components = [
        part.strip()
        for part in re.split(r"[;|]+", text)
        if part.strip()
    ]

    active_components = [
        component
        for component in components
        if not is_control_only_component(component)
    ]

    return active_components


def normalise_active_component(component):
    text = normalise_unicode_text(component).lower()

    # Remove trademark symbols.
    text = text.replace("®", "").replace("™", "")

    # Remove common administrative prefixes that do not identify the drug.
    prefix_patterns = [
        r"^treatment with\s+",
        r"^administration of\s+",
        r"^study drug\s+"
    ]

    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text)

    # Remove simple dose expressions such as 2.4 mg or 10 mg/ml.
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ug|μg|g|ml)(?:/ml)?\b",
        " ",
        text
    )

    # Remove common dosage-form / administration words. These describe
    # presentation rather than the underlying active intervention.
    form_words = [
        "injection",
        "injectable",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "oral",
        "solution",
        "infusion",
        "pen injector",
        "extended release",
        "extended-release",
        "dose tapering regimen",
        "maintenance dose regimen"
    ]

    for word in form_words:
        text = text.replace(word, " ")

    # Replace remaining punctuation with spaces and collapse whitespace.
    text = re.sub(r"[^a-z0-9+\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_intervention_key(name):
    active_components = active_components_from_name(name)

    normalised_components = [
        normalise_active_component(component)
        for component in active_components
    ]

    normalised_components = [
        component
        for component in normalised_components
        if component
    ]

    # Remove duplicate components while preserving order.
    normalised_components = list(
        dict.fromkeys(normalised_components)
    )

    return " | ".join(normalised_components)


# Start from all pharmacological intervention rows and create a
# normalised active-intervention key.
programme_interventions_df = (
    pharmacological_interventions_df[
        ["nct_id", "intervention_name"]
    ]
    .copy()
)

programme_interventions_df["intervention_key"] = (
    programme_interventions_df["intervention_name"]
    .apply(build_intervention_key)
)

# Rows with an empty key were control-only intervention entries.
control_only_row_count = (
    programme_interventions_df["intervention_key"] == ""
).sum()

programme_interventions_df = programme_interventions_df[
    programme_interventions_df["intervention_key"] != ""
].copy()

# Add the lead sponsor to every active intervention row.
programme_interventions_df = programme_interventions_df.merge(
    trials_clean_df[["nct_id", "lead_sponsor"]],
    on="nct_id",
    how="left"
)

programme_interventions_df["sponsor_key"] = (
    programme_interventions_df["lead_sponsor"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

print()
print("Control-only pharmacological intervention rows removed from programme matching:", control_only_row_count)
print(
    "Unique normalized active intervention keys:",
    programme_interventions_df["intervention_key"].nunique()
)

# Identify trials with direct weight-related primary outcomes.
direct_weight_trial_ids = set(
    trial_review_df.loc[
        trial_review_df["direct_weight_outcome_signal"],
        "nct_id"
    ]
)

# Strong seed evidence: exact sponsor + normalized active-intervention
# pairs observed in direct weight-outcome trials.
weight_programme_pairs = set(
    zip(
        programme_interventions_df.loc[
            programme_interventions_df["nct_id"].isin(
                direct_weight_trial_ids
            ),
            "sponsor_key"
        ],
        programme_interventions_df.loc[
            programme_interventions_df["nct_id"].isin(
                direct_weight_trial_ids
            ),
            "intervention_key"
        ]
    )
)

programme_interventions_df["same_programme_weight_evidence"] = [
    (sponsor, intervention) in weight_programme_pairs
    for sponsor, intervention in zip(
        programme_interventions_df["sponsor_key"],
        programme_interventions_df["intervention_key"]
    )
]

programme_signal_by_trial = (
    programme_interventions_df
    .groupby("nct_id")["same_programme_weight_evidence"]
    .any()
    .reset_index()
    .rename(
        columns={
            "same_programme_weight_evidence":
                "same_programme_weight_evidence_signal"
        }
    )
)

trial_review_df = trial_review_df.merge(
    programme_signal_by_trial,
    on="nct_id",
    how="left"
)

# NaN means there was no active programme row for the trial.
# .eq(True) converts both NaN and False to False without the
# deprecated fillna downcasting behaviour.
trial_review_df["same_programme_weight_evidence_signal"] = (
    trial_review_df["same_programme_weight_evidence_signal"]
    .eq(True)
)

# Look for language characteristic of development-enabling studies.
development_phrases = [
    "pharmacokinetic",
    "pharmacokinetics",
    "pharmacodynamic",
    "pharmacodynamics",
    "bioavailability",
    "relative bioavailability",
    "safety and tolerability",
    "safety, tolerability",
    "tolerability and pharmacokinetics",
    "single ascending dose",
    "multiple ascending dose",
    "dose escalation",
    "dose-escalation",
    "food effect",
    "formulation",
    "made available in the body",
    "maximum observed plasma concentration",
    "area under the plasma concentration",
    "aucinf",
    "cmax"
]

trial_review_df["development_context_signal"] = (
    trial_review_df["review_text"]
    .apply(
        lambda text: contains_any(
            text,
            development_phrases
        )
    )
)

trial_review_df["development_enabling_signal"] = (
    trial_review_df["same_programme_weight_evidence_signal"]
    & trial_review_df["development_context_signal"]
)

trial_review_df["positive_obesity_signal"] = (
    trial_review_df["include_signal"]
    | trial_review_df["direct_weight_outcome_signal"]
    | trial_review_df["development_enabling_signal"]
)

print()
print(
    "Sponsor-intervention programme pairs with direct weight evidence:",
    len(weight_programme_pairs)
)

print(
    "Trials matching a same-sponsor/same-intervention weight programme:",
    trial_review_df[
        "same_programme_weight_evidence_signal"
    ].sum()
)

print(
    "Development-enabling programme matches:",
    trial_review_df[
        "development_enabling_signal"
    ].sum()
)


# ============================================================
# 11. ASSIGN AUTOMATED REVIEW CATEGORY
# ============================================================

def assign_review_category(row):

    if (
        row["positive_obesity_signal"]
        and not row["exclude_signal"]
    ):
        return "LIKELY_INCLUDE"

    if (
        row["exclude_signal"]
        and not row["positive_obesity_signal"]
    ):
        return "LIKELY_EXCLUDE"

    return "MANUAL_REVIEW"


trial_review_df["review_category"] = (
    trial_review_df.apply(
        assign_review_category,
        axis=1
    )
)


# ============================================================
# 12. TRIAGE VALIDATION OUTPUT
# ============================================================

print()
print("=== Automated trial triage ===")

print(
    trial_review_df[
        "review_category"
    ]
    .value_counts()
)

triage_display_columns = [
    "nct_id",
    "brief_title",
    "pharmacological_interventions",
    "review_category"
]

for category in [
    "LIKELY_INCLUDE",
    "LIKELY_EXCLUDE",
    "MANUAL_REVIEW"
]:

    print()
    print(
        f"--- {category}: first 10 studies ---"
    )

    sample = (
        trial_review_df[
            trial_review_df[
                "review_category"
            ] == category
        ][triage_display_columns]
        .head(10)
    )

    print(
        sample.to_string(index=False)
    )


# ============================================================
# 13. MANUAL-REVIEW SAMPLE WITH PRIMARY OUTCOMES
# ============================================================

outcome_review_columns = [
    "nct_id",
    "brief_title",
    "pharmacological_interventions",
    "primary_outcome_measures",
    "review_category"
]

print()
print(
    "--- MANUAL_REVIEW with primary outcomes: first 20 ---"
)

manual_outcome_sample = (
    trial_review_df[
        trial_review_df[
            "review_category"
        ] == "MANUAL_REVIEW"
    ][outcome_review_columns]
    .head(20)
)

print(
    manual_outcome_sample.to_string(index=False)
)

# ============================================================
# 13. SAVE AUTOMATED TRIAGE AND MANUAL-REVIEW DATASETS
# ============================================================

# Preserve the automated category explicitly.
# Later, final_decision will hold our final human-reviewed decision.
trial_review_df["auto_review_category"] = (
    trial_review_df["review_category"]
)

# Columns useful for manual scientific review
manual_review_columns = [
    "nct_id",
    "brief_title",
    "official_title",
    "conditions",
    "phase",
    "primary_purpose",
    "overall_status",
    "lead_sponsor",
    "lead_sponsor_class",
    "pharmacological_interventions",
    "brief_summary",
    "primary_outcome_measures",
    "primary_outcome_descriptions",
    "include_signal",
    "exclude_signal",
    "direct_weight_outcome_signal",
    "positive_obesity_signal",
    "auto_review_category"
]

# Full audit table
triage_output_file = (
    processed_data_dir
    / "trial_inclusion_triage.csv"
)

trial_review_df.to_csv(
    triage_output_file,
    index=False
)

# Only studies that still require adjudication
manual_review_df = (
    trial_review_df[
        trial_review_df["auto_review_category"]
        == "MANUAL_REVIEW"
    ][manual_review_columns]
    .copy()
)

manual_review_file = (
    processed_data_dir
    / "manual_trial_review.csv"
)

manual_review_df.to_csv(
    manual_review_file,
    index=False
)

print()
print("=== Review files saved ===")
print("Full triage table:", triage_output_file)
print("Manual review table:", manual_review_file)
print("Manual review trials:", len(manual_review_df))
