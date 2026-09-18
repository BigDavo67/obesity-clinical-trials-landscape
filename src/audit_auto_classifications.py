from pathlib import Path
import re
import pandas as pd


# ============================================================
# PHASE 4.4E — AUDIT AUTOMATIC CLASSIFICATIONS
# Corrected version: rebuilds programme evidence directly
# from trial_inclusion_triage.csv.
# ============================================================

processed_data_dir = Path("data/processed")
triage_file = processed_data_dir / "trial_inclusion_triage.csv"

if not triage_file.exists():
    raise FileNotFoundError(
        "Could not find data/processed/trial_inclusion_triage.csv. "
        "Run clean_trials.py first."
    )

triage_df = pd.read_csv(triage_file)

print("=== Automatic classification audit ===")
print("Rows in full triage table:", len(triage_df))

if "auto_review_category" in triage_df.columns:
    category_col = "auto_review_category"
elif "review_category" in triage_df.columns:
    category_col = "review_category"
else:
    raise ValueError(
        "The triage file does not contain review_category or "
        "auto_review_category."
    )

print()
print("Automatic categories:")
print(triage_df[category_col].value_counts(dropna=False))


# ============================================================
# 1. HELPERS
# ============================================================

def bool_series(df, column_name):
    if column_name not in df.columns:
        return pd.Series(False, index=df.index)

    values = df[column_name]

    if values.dtype == bool:
        return values.fillna(False)

    return (
        values
        .fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


def combine_text(df, columns):
    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available_columns:
        return pd.Series("", index=df.index)

    return (
        df[available_columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.lower()
    )


def matched_phrases(text, phrases):
    return "; ".join(
        phrase
        for phrase in phrases
        if phrase in text
    )


def split_intervention_string(value):
    """
    Split a trial-level intervention string into individual components.
    Handles normal and full-width semicolons.
    """
    if pd.isna(value):
        return []

    text = str(value).replace("；", ";")

    return [
        piece.strip()
        for piece in text.split(";")
        if piece.strip()
    ]


def is_control_only(name):
    """
    Identify intervention components that are clearly controls rather
    than active pharmacological agents.
    """
    text = str(name).strip().lower()

    control_terms = [
        "placebo",
        "matching placebo",
        "matched placebo",
        "normal saline",
        "saline control",
        "vehicle control",
        "sham"
    ]

    # A component is control-only if it contains a control term and
    # contains no obvious named active component alongside it.
    if any(term in text for term in control_terms):
        active_words = re.sub(
            r"\b(placebo|matching|matched|normal|saline|vehicle|control|sham|"
            r"tablet|capsule|injection|solution|oral|subcutaneous|sc|iv|"
            r"intravenous|dose|doses|volume|treatment|group|arm)\b",
            " ",
            text
        )
        active_words = re.sub(r"[^a-z0-9]+", " ", active_words).strip()

        # If nothing meaningful remains, treat as control-only.
        if not active_words:
            return True

    return False


def normalize_intervention_name(name):
    """
    Conservative normalization for programme matching.
    Designed to align strings such as:
      'HRS9531 injection'
      'HRS9531 injection；Placebo'
    while avoiding aggressive synonym guessing.
    """
    text = str(name).lower().strip()

    # Remove bracketed brand/formulation notes
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\([^)]*placebo[^)]*\)", " ", text)

    # Standardise punctuation
    text = text.replace("®", " ")
    text = text.replace("™", " ")
    text = text.replace("；", ";")

    # Remove common formulation / administration words
    removable_words = [
        "injection",
        "injectable",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "solution",
        "oral",
        "subcutaneous",
        "intravenous",
        "infusion",
        "pen injector",
        "extended release",
        "extended-release",
        "treatment with",
        "study drug",
        "medication"
    ]

    for phrase in removable_words:
        text = text.replace(phrase, " ")

    # Remove common dose expressions
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|µg|ug|g|ml|mg/ml|mcg/ml)\b",
        " ",
        text
    )

    # Remove placebo/control wording if mixed with an active name
    text = re.sub(
        r"\b(?:matching|matched)?\s*placebo\b",
        " ",
        text
    )
    text = re.sub(r"\bnormal saline\b", " ", text)

    # Keep alphanumerics and useful separators
    text = re.sub(r"[^a-z0-9+\-/ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ============================================================
# 2. BUILD AUDIT TEXT
# ============================================================

audit_text_columns = [
    "brief_title",
    "official_title",
    "conditions",
    "brief_summary",
    "pharmacological_interventions",
    "primary_outcome_measures",
    "primary_outcome_descriptions"
]

triage_df["audit_text"] = combine_text(
    triage_df,
    audit_text_columns
)

direct_weight_signal = bool_series(
    triage_df,
    "direct_weight_outcome_signal"
)


# ============================================================
# 3. REBUILD SPONSOR–INTERVENTION PROGRAMME EVIDENCE
# ============================================================

required_columns = [
    "nct_id",
    "lead_sponsor",
    "pharmacological_interventions"
]

missing_required = [
    column
    for column in required_columns
    if column not in triage_df.columns
]

if missing_required:
    raise ValueError(
        "Missing required columns for programme matching: "
        + ", ".join(missing_required)
    )

programme_rows = []

for _, row in triage_df.iterrows():

    nct_id = row["nct_id"]
    sponsor = str(row["lead_sponsor"]).strip().lower()

    intervention_components = split_intervention_string(
        row["pharmacological_interventions"]
    )

    for component in intervention_components:

        if is_control_only(component):
            continue

        normalized_key = normalize_intervention_name(component)

        if not normalized_key:
            continue

        programme_rows.append(
            {
                "nct_id": nct_id,
                "lead_sponsor_normalized": sponsor,
                "active_intervention_key": normalized_key
            }
        )

programme_df = (
    pd.DataFrame(programme_rows)
    .drop_duplicates()
)

# Attach direct-weight evidence to each trial-programme pair
programme_df = programme_df.merge(
    triage_df[
        ["nct_id", "direct_weight_outcome_signal"]
    ],
    on="nct_id",
    how="left"
)

programme_df["direct_weight_outcome_signal"] = (
    programme_df["direct_weight_outcome_signal"]
    .fillna(False)
    .astype(str)
    .str.lower()
    .isin(["true", "1", "yes"])
)

# Programme pairs that appear in at least one direct-weight trial
weight_programme_pairs = (
    programme_df[
        programme_df["direct_weight_outcome_signal"]
    ][
        [
            "lead_sponsor_normalized",
            "active_intervention_key"
        ]
    ]
    .drop_duplicates()
)

weight_programme_pairs["programme_weight_evidence"] = True

programme_df = programme_df.merge(
    weight_programme_pairs,
    on=[
        "lead_sponsor_normalized",
        "active_intervention_key"
    ],
    how="left"
)

programme_df["programme_weight_evidence"] = (
    programme_df["programme_weight_evidence"]
    .fillna(False)
    .astype(bool)
)

# Trial-level programme evidence
trial_programme_signal = (
    programme_df
    .groupby("nct_id")["programme_weight_evidence"]
    .max()
)

triage_df["programme_weight_evidence"] = (
    triage_df["nct_id"]
    .map(trial_programme_signal)
    .fillna(False)
    .astype(bool)
)

programme_signal = triage_df["programme_weight_evidence"]


# ============================================================
# 4. SUSPICIOUS-CONTEXT FLAGS FOR AUTO-INCLUDES
# ============================================================

suspicious_include_phrases = [
    "aging",
    "ageing",
    "longevity",
    "lifespan",
    "healthspan",

    "pregnan",
    "gestational",
    "fetal",
    "foetal",
    "maternal",
    "birth weight",
    "preeclampsia",
    "pre-eclampsia",

    "atrial fibrillation",
    "heart failure",
    "hypertension",
    "blood pressure",
    "arterial stiffness",
    "vascular",
    "myocardial",
    "coronary",

    "renal",
    "kidney",
    "glomerular",

    "fibrosis",
    "fatty liver",
    "nafld",
    "nash",

    "hba1c",
    "glycemic",
    "glycaemic",
    "glucose control",

    "nausea",
    "vomiting",
    "antiemetic",
    "pain",
    "analges",
    "anesthesia",
    "anaesthesia",

    "fertility",
    "ovulation",
    "contraception",

    "mechanistic",
    "mechanism study",
    "brain response",
    "food cue",
    "energy expenditure",
    "brown adipose",
    "thermogenesis"
]

triage_df["suspicious_context_matches"] = (
    triage_df["audit_text"]
    .apply(
        lambda text: matched_phrases(
            text,
            suspicious_include_phrases
        )
    )
)

suspicious_context_signal = (
    triage_df["suspicious_context_matches"] != ""
)


# ============================================================
# 5. SELECT AUTOMATIC INCLUDES FOR AUDIT
# ============================================================

auto_include_mask = (
    triage_df[category_col] == "LIKELY_INCLUDE"
)

strong_include_evidence = (
    direct_weight_signal
    | programme_signal
)

# Review auto-includes with neither strong evidence source.
weak_auto_include_mask = (
    auto_include_mask
    & ~strong_include_evidence
)

# Also review suspicious-context auto-includes even if programme
# evidence exists, unless they have a direct weight primary outcome.
#
# This catches cases such as a known obesity drug being used for
# another indication.
suspicious_auto_include_mask = (
    auto_include_mask
    & suspicious_context_signal
    & ~direct_weight_signal
)

auto_include_audit_mask = (
    weak_auto_include_mask
    | suspicious_auto_include_mask
)


# ============================================================
# 6. AUDIT ALL AUTOMATIC EXCLUDES
# ============================================================

auto_exclude_audit_mask = (
    triage_df[category_col] == "LIKELY_EXCLUDE"
)


# ============================================================
# 7. CREATE AUDIT REASONS
# ============================================================

triage_df["audit_reason"] = ""

triage_df.loc[
    weak_auto_include_mask,
    "audit_reason"
] = (
    "AUTO_INCLUDE without direct weight primary-outcome evidence "
    "or reconstructed same-programme weight evidence"
)

context_reason_mask = suspicious_auto_include_mask

triage_df.loc[
    context_reason_mask,
    "audit_reason"
] = (
    triage_df.loc[
        context_reason_mask,
        "audit_reason"
    ]
    + " | suspicious context: "
    + triage_df.loc[
        context_reason_mask,
        "suspicious_context_matches"
    ]
)

triage_df.loc[
    auto_exclude_audit_mask,
    "audit_reason"
] = (
    "All AUTO_EXCLUDE trials are sanity-audited for false negatives"
)


# ============================================================
# 8. EXPORT AUDIT TABLES
# ============================================================

preferred_columns = [
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
    "programme_weight_evidence",
    category_col,
    "suspicious_context_matches",
    "audit_reason"
]

audit_columns = [
    column
    for column in preferred_columns
    if column in triage_df.columns
]

auto_include_audit_df = (
    triage_df[
        auto_include_audit_mask
    ][audit_columns]
    .copy()
)

auto_exclude_audit_df = (
    triage_df[
        auto_exclude_audit_mask
    ][audit_columns]
    .copy()
)

auto_audit_df = pd.concat(
    [
        auto_include_audit_df,
        auto_exclude_audit_df
    ],
    ignore_index=True
)

auto_include_file = (
    processed_data_dir
    / "auto_include_audit.csv"
)

auto_exclude_file = (
    processed_data_dir
    / "auto_exclude_audit.csv"
)

combined_audit_file = (
    processed_data_dir
    / "auto_classification_audit.csv"
)

auto_include_audit_df.to_csv(
    auto_include_file,
    index=False
)

auto_exclude_audit_df.to_csv(
    auto_exclude_file,
    index=False
)

auto_audit_df.to_csv(
    combined_audit_file,
    index=False
)


# ============================================================
# 9. VALIDATION OUTPUT
# ============================================================

print()
print("=== Reconstructed programme evidence ===")

print(
    "Unique sponsor-intervention programme pairs:",
    programme_df[
        [
            "lead_sponsor_normalized",
            "active_intervention_key"
        ]
    ].drop_duplicates().shape[0]
)

print(
    "Programme pairs with direct weight evidence:",
    len(weight_programme_pairs)
)

print(
    "Trials matching a same-sponsor/same-intervention "
    "weight programme:",
    programme_signal.sum()
)

print()
print("=== Audit subset ===")

print(
    "Automatic LIKELY_INCLUDE trials:",
    auto_include_mask.sum()
)

print(
    "Auto-includes with direct weight outcome evidence:",
    (auto_include_mask & direct_weight_signal).sum()
)

print(
    "Auto-includes with reconstructed programme evidence:",
    (auto_include_mask & programme_signal).sum()
)

print(
    "Auto-includes selected for sanity audit:",
    len(auto_include_audit_df)
)

print(
    "Automatic LIKELY_EXCLUDE trials:",
    auto_exclude_audit_mask.sum()
)

print(
    "Auto-excludes selected for sanity audit:",
    len(auto_exclude_audit_df)
)

print(
    "Total trials requiring automatic-classification audit:",
    len(auto_audit_df)
)

print()
print("Saved:")
print(auto_include_file)
print(auto_exclude_file)
print(combined_audit_file)

print()
print("--- First 15 auto-include audit cases ---")

display_columns = [
    "nct_id",
    "brief_title",
    "pharmacological_interventions",
    "direct_weight_outcome_signal",
    "programme_weight_evidence",
    "suspicious_context_matches",
    "audit_reason"
]

print(
    auto_include_audit_df[
        display_columns
    ]
    .head(15)
    .to_string(index=False)
)
