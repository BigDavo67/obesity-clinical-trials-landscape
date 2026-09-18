from pathlib import Path
import re
import pandas as pd


# ============================================================
# PHASE 4.6B — TRIAL-SPECIFIC INTERVENTION ROLE CLASSIFICATION
#
# Roles:
#   FOCAL_OBESITY_THERAPY
#   ACTIVE_COMPARATOR
#   PK_DDI_PROBE
#   POSITIVE_CONTROL
#   BACKGROUND_OR_CONCOMITANT
#   REVIEW_REQUIRED
#
# This script is deliberately conservative. It only auto-assigns
# focal/comparator roles when there is strong trial-level evidence.
# ============================================================

processed_data_dir = Path("data/processed")

input_file = (
    processed_data_dir
    / "phase46_trial_intervention_review.csv"
)

df = pd.read_csv(input_file)

print("=== Phase 4.6B: trial-specific intervention roles ===")
print("Trial-intervention rows:", len(df))
print("Trials:", df["nct_id"].nunique())


# ============================================================
# 1. HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()

    text = (
        text
        .replace("®", " ")
        .replace("™", " ")
        .replace("–", "-")
        .replace("—", "-")
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def contains_any(text, phrases):
    text = clean_text(text)

    return any(
        phrase in text
        for phrase in phrases
    )


def normalize_analysis_name(row):
    """
    Small final cleanup layer for obvious wording variants that
    do not need scientific alias research.

    We do NOT guess unknown development-code aliases here.
    """
    canonical = clean_text(
        row.get("canonical_candidate", "")
    )

    key = clean_text(
        row.get("active_name_key", "")
    )

    name = canonical or key

    # Common generic/formulation wording
    if "semaglutide" in name and not any(
        token in name
        for token in [
            "cagrilintide",
            "tirzepatide",
            "metformin",
            "and ",
            " + "
        ]
    ):
        return "semaglutide"

    if "tirzepatide" in name and not any(
        token in name
        for token in [
            "eloralintide",
            "semaglutide",
            "and ",
            " + "
        ]
    ):
        return "tirzepatide"

    if "liraglutide" in name:
        return "liraglutide"

    if "phentermine" in name and "topiramate" not in name:
        return "phentermine"

    if "oxytocin" in name and "nasal" in name:
        return "oxytocin"

    if "empagliflozin" in name:
        return "empagliflozin"

    if "efsubaglutide alfa" in name:
        return "efsubaglutide alfa"

    if "lisdexamfetamine" in name:
        return "lisdexamfetamine"

    # Specific combination text variants
    if (
        "cagrilintide" in name
        and "semaglutide" in name
    ):
        return "cagrilintide + semaglutide"

    if (
        "eloralintide" in name
        and "tirzepatide" in name
    ):
        return "eloralintide + tirzepatide"

    return name


df["analysis_name_candidate"] = (
    df.apply(
        normalize_analysis_name,
        axis=1
    )
)


# ============================================================
# 2. BUILD ROLE-SPECIFIC TEXT FIELDS
# ============================================================

source_columns = [
    "active_component_source",
    "active_component_raw",
    "intervention_description",
    "arm_group_labels",
]

trial_columns = [
    "brief_title",
    "official_title",
    "brief_summary",
    "primary_outcome_measures",
    "primary_outcome_descriptions",
]

def combine_columns(row, columns):
    values = []

    for column in columns:
        if column in row.index:
            value = row[column]

            if pd.notna(value):
                values.append(
                    str(value)
                )

    return clean_text(
        " ".join(values)
    )


df["intervention_role_text"] = (
    df.apply(
        lambda row: combine_columns(
            row,
            source_columns
        ),
        axis=1
    )
)

df["trial_role_text"] = (
    df.apply(
        lambda row: combine_columns(
            row,
            trial_columns
        ),
        axis=1
    )
)


# ============================================================
# 3. HIGH-CONFIDENCE NON-FOCAL DRUGS
# ============================================================

pk_ddi_probe_names = {
    "acetaminophen",
    "paracetamol",
    "digoxin",
    "midazolam",
    "warfarin",
    "rosuvastatin",
    "atorvastatin",
    "caffeine",
    "omeprazole",
    "dextromethorphan",
}

positive_control_names = {
    "moxifloxacin",
}

background_name_patterns = [
    "contraceptive",
    "ethinyl estradiol",
    "levonorgestrel",
]


# ============================================================
# 4. EXPLICIT ROLE LANGUAGE
# ============================================================

comparator_phrases = [
    "active comparator",
    "comparator arm",
    "comparator group",
    "reference treatment",
    "reference drug",
    "positive comparator",
]

background_phrases = [
    "background therapy",
    "background medication",
    "concomitant medication",
    "concomitant therapy",
    "standard of care",
    "stable dose",
    "continued treatment",
    "continue treatment",
    "if taken before",
    "rescue medication",
]

probe_phrases = [
    "probe substrate",
    "drug-drug interaction",
    "drug drug interaction",
    "ddi study",
    "cyp cocktail",
    "cocktail substrate",
    "pharmacokinetic interaction",
]

qt_phrases = [
    "thorough qt",
    "qt interval",
    "qtc",
]


# ============================================================
# 5. FIRST-PASS ROLE ASSIGNMENT
# ============================================================

df["role_candidate"] = "REVIEW_REQUIRED"
df["role_confidence"] = "low"
df["role_reason"] = ""


for idx, row in df.iterrows():

    name = clean_text(
        row["analysis_name_candidate"]
    )

    intervention_text = row["intervention_role_text"]
    trial_text = row["trial_role_text"]

    # --------------------------------------------------------
    # 5A. Positive control
    # --------------------------------------------------------

    if (
        name in positive_control_names
        and contains_any(
            trial_text,
            qt_phrases
        )
    ):
        df.at[idx, "role_candidate"] = "POSITIVE_CONTROL"
        df.at[idx, "role_confidence"] = "high"
        df.at[idx, "role_reason"] = (
            "Known QT positive-control drug in a QT/QTc study."
        )
        continue

    # --------------------------------------------------------
    # 5B. PK / DDI probe
    # --------------------------------------------------------

    if name in pk_ddi_probe_names:
        df.at[idx, "role_candidate"] = "PK_DDI_PROBE"
        df.at[idx, "role_confidence"] = "high"
        df.at[idx, "role_reason"] = (
            "Known PK/DDI probe or marker drug in obesity-drug "
            "development studies."
        )
        continue

    if (
        contains_any(
            intervention_text,
            probe_phrases
        )
        and name not in {
            "",
            "semaglutide",
            "tirzepatide",
            "liraglutide",
        }
    ):
        df.at[idx, "role_candidate"] = "PK_DDI_PROBE"
        df.at[idx, "role_confidence"] = "medium"
        df.at[idx, "role_reason"] = (
            "Intervention-specific wording identifies a PK/DDI role."
        )
        continue

    # --------------------------------------------------------
    # 5C. Background / concomitant
    # --------------------------------------------------------

    if any(
        phrase in name
        for phrase in background_name_patterns
    ):
        df.at[idx, "role_candidate"] = "BACKGROUND_OR_CONCOMITANT"
        df.at[idx, "role_confidence"] = "high"
        df.at[idx, "role_reason"] = (
            "Contraceptive/background medication rather than obesity therapy."
        )
        continue

    if contains_any(
        intervention_text,
        background_phrases
    ):
        df.at[idx, "role_candidate"] = "BACKGROUND_OR_CONCOMITANT"
        df.at[idx, "role_confidence"] = "high"
        df.at[idx, "role_reason"] = (
            "Intervention-specific wording explicitly describes background/"
            "concomitant treatment."
        )
        continue

    # --------------------------------------------------------
    # 5D. Explicit active comparator
    # --------------------------------------------------------

    if contains_any(
        intervention_text,
        comparator_phrases
    ):
        df.at[idx, "role_candidate"] = "ACTIVE_COMPARATOR"
        df.at[idx, "role_confidence"] = "high"
        df.at[idx, "role_reason"] = (
            "Intervention/arm wording explicitly identifies an active comparator."
        )
        continue


# ============================================================
# 6. COUNT REMAINING CANDIDATE THERAPIES PER TRIAL
# ============================================================

non_focal_roles = {
    "PK_DDI_PROBE",
    "POSITIVE_CONTROL",
    "BACKGROUND_OR_CONCOMITANT",
}

df["is_remaining_therapy_candidate"] = (
    ~df["role_candidate"].isin(
        non_focal_roles
    )
)

remaining_counts = (
    df[
        df["is_remaining_therapy_candidate"]
    ]
    .groupby("nct_id")[
        "analysis_name_candidate"
    ]
    .nunique()
)

df["remaining_therapy_count"] = (
    df["nct_id"]
    .map(remaining_counts)
    .fillna(0)
    .astype(int)
)


# ============================================================
# 7. HIGH-CONFIDENCE FOCAL THERAPY RULES
# ============================================================

def name_in_text(name, text):
    name = clean_text(name)
    text = clean_text(text)

    if not name or not text:
        return False

    # For combinations, require at least the major component tokens.
    if " + " in name:
        components = [
            component.strip()
            for component in name.split("+")
            if component.strip()
        ]

        return all(
            component in text
            for component in components
        )

    return name in text


for idx, row in df.iterrows():

    if row["role_candidate"] != "REVIEW_REQUIRED":
        continue

    name = row["analysis_name_candidate"]
    trial_text = row["trial_role_text"]
    count = row["remaining_therapy_count"]

    # --------------------------------------------------------
    # 7A. Only one remaining pharmacological therapy identity
    # --------------------------------------------------------

    if count == 1:
        df.at[idx, "role_candidate"] = "FOCAL_OBESITY_THERAPY"
        df.at[idx, "role_confidence"] = "high"
        df.at[idx, "role_reason"] = (
            "Only remaining active pharmacological therapy identity "
            "after probes/background treatments were removed."
        )
        continue

    # --------------------------------------------------------
    # 7B. Multi-drug trial, but therapy is explicitly named in title/
    #     summary/outcomes.
    #
    # Keep as medium confidence because another active comparator
    # may also be present.
    # --------------------------------------------------------

    if (
        count > 1
        and name_in_text(
            name,
            trial_text
        )
    ):
        df.at[idx, "role_candidate"] = "FOCAL_OBESITY_THERAPY"
        df.at[idx, "role_confidence"] = "medium"
        df.at[idx, "role_reason"] = (
            "Therapy is explicitly named in the study title/summary/outcome "
            "context in a multi-active-intervention trial."
        )


# ============================================================
# 8. EXPLICIT TRIAL-LEVEL COMPARATOR LANGUAGE
# ============================================================

comparison_connectors = [
    " versus ",
    " vs ",
    " vs. ",
    " compared with ",
    " compared to ",
    " comparison with ",
]

for nct_id, group in df.groupby("nct_id"):

    unresolved_indices = group.index[
        group["role_candidate"] == "REVIEW_REQUIRED"
    ].tolist()

    if not unresolved_indices:
        continue

    title_text = clean_text(
        " ".join(
            str(value)
            for value in group[
                [
                    column
                    for column in [
                        "brief_title",
                        "official_title"
                    ]
                    if column in group.columns
                ]
            ]
            .iloc[0]
            .dropna()
            .tolist()
        )
    )

    if not contains_any(
        title_text,
        comparison_connectors
    ):
        continue

    for idx in unresolved_indices:

        name = clean_text(
            df.at[
                idx,
                "analysis_name_candidate"
            ]
        )

        # If a drug appears after an explicit comparison connector,
        # flag it as a likely active comparator.
        comparator_signal = any(
            connector + name in title_text
            for connector in comparison_connectors
            if name
        )

        if comparator_signal:
            df.at[idx, "role_candidate"] = "ACTIVE_COMPARATOR"
            df.at[idx, "role_confidence"] = "medium"
            df.at[idx, "role_reason"] = (
                "Drug name appears after explicit versus/compared-with "
                "language in the trial title."
            )


# ============================================================
# 9. FINAL REVIEW SUBSET
# ============================================================

manual_review_df = (
    df[
        df["role_candidate"] == "REVIEW_REQUIRED"
    ]
    .copy()
)

# Prioritize high-impact unresolved roles.
trial_frequency = (
    df.groupby(
        "analysis_name_candidate"
    )["nct_id"]
    .nunique()
)

manual_review_df[
    "therapy_trial_frequency"
] = (
    manual_review_df[
        "analysis_name_candidate"
    ]
    .map(trial_frequency)
    .fillna(0)
    .astype(int)
)

manual_review_df = (
    manual_review_df
    .sort_values(
        by=[
            "therapy_trial_frequency",
            "nct_id"
        ],
        ascending=[
            False,
            True
        ]
    )
)


# ============================================================
# 10. SAVE
# ============================================================

all_roles_file = (
    processed_data_dir
    / "phase46_intervention_role_candidates.csv"
)

manual_roles_file = (
    processed_data_dir
    / "phase46_intervention_role_manual_review.csv"
)

df.to_csv(
    all_roles_file,
    index=False
)

manual_review_df.to_csv(
    manual_roles_file,
    index=False
)


# ============================================================
# 11. VALIDATION OUTPUT
# ============================================================

print()
print("=== Role classification result ===")
print(
    df["role_candidate"]
    .value_counts(dropna=False)
)

print()
print("=== Confidence ===")
print(
    df["role_confidence"]
    .value_counts(dropna=False)
)

print()
print(
    "Rows still requiring role review:",
    len(manual_review_df)
)

print(
    "Trials containing at least one unresolved role:",
    manual_review_df["nct_id"].nunique()
)

print()
print("--- Top 30 unresolved role rows ---")

display_columns = [
    column
    for column in [
        "nct_id",
        "brief_title",
        "analysis_name_candidate",
        "active_component_source",
        "arm_group_labels",
        "remaining_therapy_count",
        "therapy_trial_frequency"
    ]
    if column in manual_review_df.columns
]

print(
    manual_review_df[
        display_columns
    ]
    .head(30)
    .to_string(index=False)
)

print()
print("--- Top 20 therapy identities by focal assignments ---")

focal_summary = (
    df[
        df["role_candidate"]
        == "FOCAL_OBESITY_THERAPY"
    ]
    .groupby(
        "analysis_name_candidate"
    )
    .agg(
        focal_trials=(
            "nct_id",
            "nunique"
        ),
        focal_rows=(
            "nct_id",
            "size"
        )
    )
    .reset_index()
    .sort_values(
        "focal_trials",
        ascending=False
    )
)

print(
    focal_summary
    .head(20)
    .to_string(index=False)
)

print()
print("Saved:")
print(all_roles_file)
print(manual_roles_file)
