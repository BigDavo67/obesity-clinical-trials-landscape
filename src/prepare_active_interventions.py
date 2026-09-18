from pathlib import Path
import re
import pandas as pd


# ============================================================
# PHASE 4.5 — ACTIVE PHARMACOLOGICAL INTERVENTIONS
# Final parser: preserves active drug from mixed strings such as
# "ISM8969 tablets or placebo"
# ============================================================

processed_data_dir = Path("data/processed")

trials_file = processed_data_dir / "obesity_trials_clean.csv"
interventions_file = processed_data_dir / "obesity_interventions_clean.csv"

trials_df = pd.read_csv(trials_file)
interventions_df = pd.read_csv(interventions_file)

print("=== Phase 4.5: active pharmacological interventions ===")
print("Included trials:", len(trials_df))
print("Intervention rows:", len(interventions_df))


# ============================================================
# 1. RELATIONAL VALIDATION
# ============================================================

included_trial_ids = set(trials_df["nct_id"])
intervention_trial_ids = set(interventions_df["nct_id"])

orphan_ids = intervention_trial_ids - included_trial_ids

if orphan_ids:
    raise ValueError(
        f"Found {len(orphan_ids)} orphan intervention trial IDs."
    )

print("Orphan intervention IDs:", len(orphan_ids))


# ============================================================
# 2. KEEP PHARMACOLOGICAL INTERVENTION TYPES
# ============================================================

pharmacological_types = [
    "DRUG",
    "BIOLOGICAL",
    "COMBINATION_PRODUCT"
]

pharma_df = interventions_df[
    interventions_df["intervention_type"].isin(
        pharmacological_types
    )
].copy()

print()
print("Pharmacological rows before control removal:", len(pharma_df))

print()
print("Pharmacological intervention types:")
print(
    pharma_df["intervention_type"]
    .value_counts(dropna=False)
)


# ============================================================
# 3. SPLIT SEMICOLON-DELIMITED COMPONENTS
# ============================================================

def split_components(value):
    """
    Split source strings only on semicolons.

    We do NOT generally split on 'and', because genuine
    combinations can contain 'and'.
    """
    if pd.isna(value):
        return []

    text = str(value).replace("；", ";")

    return [
        component.strip()
        for component in text.split(";")
        if component.strip()
    ]


# ============================================================
# 4. EXTRACT ACTIVE DRUG FROM EACH COMPONENT
# ============================================================

def extract_active_component(component):
    """
    Return the active-drug portion of a registry component.

    Examples
    --------
    "Placebo"                           -> None
    "Eloralintide Placebo"              -> None
    "Placebo matched to semaglutide"    -> None
    "ISM8969 tablets or placebo"        -> "ISM8969 tablets"
    "HM15211 or Placebo"                -> "HM15211"
    "aleniglipron or placebo"           -> "aleniglipron"
    "HDM1005 injection or placebo"      -> "HDM1005 injection"
    "Drug X + placebo"                  -> "Drug X"
    "UBT251 2 mg and UBT251 Placebo"    -> "UBT251 2 mg"

    The goal is to remove the CONTROL PORTION, not the entire
    source string, when the active and placebo are combined in
    the same ClinicalTrials.gov intervention label.
    """
    if component is None:
        return None

    text = str(component).strip()

    if not text:
        return None

    lower = text.lower().strip()

    # --------------------------------------------------------
    # 4A. Pure control labels
    # --------------------------------------------------------

    pure_control_patterns = [
        r"^placebo\b",
        r"^matching placebo\b",
        r"^matched placebo\b",
        r"^placebo matched\b",
        r"^treatment with placebo\b",
        r"^normal saline\b",
        r"^saline placebo\b",
        r"^saline control\b",
        r"^vehicle control\b",
        r"^vehicle placebo\b",
        r"^sham\b",
    ]

    if any(
        re.search(pattern, lower)
        for pattern in pure_control_patterns
    ):
        return None

    exact_controls = {
        "saline",
        "normal saline",
        "vehicle",
        "control",
        "placebo oral tablet",
        "placebo injection",
    }

    if lower in exact_controls:
        return None

    # --------------------------------------------------------
    # 4B. Active drug explicitly contrasted with placebo
    #
    # "ISM8969 tablets or placebo"
    # "CPX101 or placebo 120mg Q2W"
    # "HM15211 versus placebo"
    # --------------------------------------------------------

    connector_patterns = [
        r"^(.*?)\s+\bor\b\s+(?:matching\s+|matched\s+)?placebo\b.*$",
        r"^(.*?)\s+\bversus\b\s+(?:matching\s+|matched\s+)?placebo\b.*$",
        r"^(.*?)\s+\bvs\.?\s+(?:matching\s+|matched\s+)?placebo\b.*$",
        r"^(.*?)\s*\+\s*(?:matching\s+|matched\s+)?placebo\b.*$",
    ]

    for pattern in connector_patterns:
        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            active = match.group(1).strip(" ,;/+-")

            return active if active else None

    # --------------------------------------------------------
    # 4C. Registry arm label containing:
    #     active treatment AND active-treatment placebo
    #
    # Example:
    # "UBT251 Injection 2.0 mg and UBT251 Injection Placebo"
    #
    # Only use this special rule when the second half contains
    # placebo. We do NOT split ordinary drug combinations.
    # --------------------------------------------------------

    and_placebo_match = re.match(
        r"^(.*?)\s+\band\b\s+(.+\bplacebo\b.*)$",
        text,
        flags=re.IGNORECASE
    )

    if and_placebo_match:
        active = and_placebo_match.group(1).strip(" ,;/+-")

        return active if active else None

    # --------------------------------------------------------
    # 4D. Remaining labels that contain "placebo"
    #
    # e.g. "Eloralintide Placebo", "HRS9531 injection placebo"
    # These are placebo-arm labels, not active treatment labels.
    # --------------------------------------------------------

    if re.search(
        r"\bplacebo\b",
        lower
    ):
        return None

    # --------------------------------------------------------
    # 4E. Other pure controls
    # --------------------------------------------------------

    if re.match(
        r"^(normal )?saline\b",
        lower
    ):
        return None

    if re.match(
        r"^vehicle\b",
        lower
    ):
        return None

    if re.match(
        r"^sham\b",
        lower
    ):
        return None

    return text


# ============================================================
# 5. EXPLODE TO ONE ACTIVE COMPONENT PER ROW
# ============================================================

active_rows = []

control_only_components_removed = 0
mixed_active_placebo_components_recovered = 0

for _, row in pharma_df.iterrows():

    components = split_components(
        row["intervention_name"]
    )

    for component in components:

        active_component = extract_active_component(
            component
        )

        if active_component is None:
            control_only_components_removed += 1
            continue

        if (
            re.search(
                r"\bplacebo\b",
                str(component),
                flags=re.IGNORECASE
            )
            and not re.search(
                r"\bplacebo\b",
                str(active_component),
                flags=re.IGNORECASE
            )
        ):
            mixed_active_placebo_components_recovered += 1

        active_row = row.to_dict()

        # Keep both for auditability
        active_row["active_component_source"] = component
        active_row["active_component_raw"] = active_component

        active_rows.append(active_row)

active_df = pd.DataFrame(active_rows)

print()
print("Active pharmacological component rows:", len(active_df))
print(
    "Control-only components removed:",
    control_only_components_removed
)
print(
    "Mixed active/placebo components recovered:",
    mixed_active_placebo_components_recovered
)


# ============================================================
# 6. SAFE BASIC NAME NORMALISATION
# ============================================================

def remove_phrase_as_words(text, phrase):

    pattern = (
        r"(?<![a-z0-9])"
        + re.escape(phrase)
        + r"(?![a-z0-9])"
    )

    return re.sub(
        pattern,
        " ",
        text,
        flags=re.IGNORECASE
    )


def normalize_active_name(name):
    """
    Conservative formatting normalization only.

    Scientific alias reconciliation happens in Phase 4.6.
    """
    text = str(name).strip().lower()

    text = (
        text
        .replace("®", " ")
        .replace("™", " ")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )

    # Remove square-bracket notes such as [Ozempic]
    text = re.sub(
        r"\[[^\]]*\]",
        " ",
        text
    )

    # Remove dose expressions
    text = re.sub(
        r"(?<![a-z0-9])"
        r"\d+(?:\.\d+)?\s*"
        r"(?:mg/ml|mcg/ml|µg/ml|ug/ml|mg|mcg|µg|ug|g|ml)"
        r"(?![a-z0-9])",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove frequency/dosing schedule tokens that often remain
    text = re.sub(
        r"(?<![a-z0-9])q\d+[wdh](?![a-z0-9])",
        " ",
        text,
        flags=re.IGNORECASE
    )

    removable_phrases = [
        "extended-release tablets",
        "extended release tablets",
        "extended-release tablet",
        "extended release tablet",
        "extended-release capsules",
        "extended release capsules",
        "extended-release capsule",
        "extended release capsule",
        "pen injector",
        "oral tablet",
        "oral tablets",
        "oral capsule",
        "oral capsules",
        "subcutaneous injection",
        "intravenous injection",
        "treatment with",
        "study drug",
        "injection",
        "injectable",
        "infusion",
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "solution",
        "oral",
        "subcutaneous",
        "intravenous",
        "single dose",
        "multiple dose",
        "medication",
    ]

    for phrase in removable_phrases:
        text = remove_phrase_as_words(
            text,
            phrase
        )

    text = re.sub(
        r"[^a-z0-9+\-/ ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


active_df["active_name_key"] = (
    active_df["active_component_raw"]
    .apply(normalize_active_name)
)

active_df = active_df[
    active_df["active_name_key"].notna()
    & (active_df["active_name_key"] != "")
].copy()


# ============================================================
# 7. ADD SPONSOR INFORMATION
# ============================================================

sponsor_columns = [
    "nct_id",
    "lead_sponsor",
    "lead_sponsor_class"
]

sponsor_df = (
    trials_df[sponsor_columns]
    .drop_duplicates(subset=["nct_id"])
)

for column in [
    "lead_sponsor",
    "lead_sponsor_class"
]:
    if column in active_df.columns:
        active_df = active_df.drop(
            columns=[column]
        )

active_df = active_df.merge(
    sponsor_df,
    on="nct_id",
    how="left"
)


# ============================================================
# 8. VALIDATE EVERY INCLUDED TRIAL HAS ACTIVE PHARMA
# ============================================================

represented_trial_ids = set(
    active_df["nct_id"]
)

missing_active_trial_ids = (
    included_trial_ids
    - represented_trial_ids
)

missing_active_trials_df = (
    trials_df[
        trials_df["nct_id"].isin(
            missing_active_trial_ids
        )
    ]
    .copy()
)

missing_interventions_df = (
    interventions_df[
        interventions_df["nct_id"].isin(
            missing_active_trial_ids
        )
    ]
    .copy()
)

if len(missing_active_trials_df) > 0:

    missing_summary = (
        missing_interventions_df
        .groupby("nct_id")
        .agg(
            source_intervention_types=(
                "intervention_type",
                lambda values: "; ".join(
                    dict.fromkeys(
                        str(v)
                        for v in values.dropna()
                    )
                )
            ),
            source_intervention_names=(
                "intervention_name",
                lambda values: "; ".join(
                    dict.fromkeys(
                        str(v)
                        for v in values.dropna()
                    )
                )
            )
        )
        .reset_index()
    )

    missing_active_trials_df = (
        missing_active_trials_df
        .merge(
            missing_summary,
            on="nct_id",
            how="left"
        )
    )


# ============================================================
# 9. BUILD UNIQUE-NAME REVIEW TABLE
# ============================================================

def join_examples(values, max_examples=5):

    unique_values = []

    for value in values.dropna():

        value = str(value).strip()

        if value and value not in unique_values:
            unique_values.append(value)

        if len(unique_values) >= max_examples:
            break

    return "; ".join(unique_values)


def join_unique(values, max_items=8):

    unique_values = []

    for value in values.dropna():

        value = str(value).strip()

        if value and value not in unique_values:
            unique_values.append(value)

        if len(unique_values) >= max_items:
            break

    return "; ".join(unique_values)


name_review_df = (
    active_df
    .groupby("active_name_key")
    .agg(
        example_raw_names=(
            "active_component_raw",
            join_examples
        ),
        example_source_labels=(
            "active_component_source",
            join_examples
        ),
        trial_count=(
            "nct_id",
            "nunique"
        ),
        row_count=(
            "nct_id",
            "size"
        ),
        intervention_types=(
            "intervention_type",
            join_unique
        ),
        example_sponsors=(
            "lead_sponsor",
            join_unique
        )
    )
    .reset_index()
)

name_review_df = (
    name_review_df
    .sort_values(
        by=[
            "trial_count",
            "active_name_key"
        ],
        ascending=[
            False,
            True
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 10. SAVE
# ============================================================

active_file = (
    processed_data_dir
    / "obesity_active_interventions_stage1.csv"
)

review_file = (
    processed_data_dir
    / "intervention_name_review.csv"
)

missing_file = (
    processed_data_dir
    / "included_trials_without_active_pharma.csv"
)

active_df.to_csv(
    active_file,
    index=False
)

name_review_df.to_csv(
    review_file,
    index=False
)

missing_active_trials_df.to_csv(
    missing_file,
    index=False
)


# ============================================================
# 11. VALIDATION OUTPUT
# ============================================================

print()
print("=== Final Phase 4.5 summary ===")

print(
    "Included trials represented:",
    active_df["nct_id"].nunique()
)

print(
    "Included trials with no retained active component:",
    len(missing_active_trial_ids)
)

print(
    "Unique raw active components:",
    active_df["active_component_raw"].nunique()
)

print(
    "Unique safe normalized keys:",
    active_df["active_name_key"].nunique()
)

print()
print("Placebo leakage check:")

placebo_leak = active_df[
    active_df["active_component_raw"]
    .astype(str)
    .str.contains(
        r"\bplacebo\b",
        case=False,
        na=False,
        regex=True
    )
]

print(
    "Active rows still containing 'placebo':",
    len(placebo_leak)
)

print()
print("Recovered mixed active/placebo examples:")

recovered_examples = (
    active_df[
        active_df["active_component_source"]
        .astype(str)
        .str.contains(
            r"\bplacebo\b",
            case=False,
            na=False,
            regex=True
        )
    ][
        [
            "active_component_source",
            "active_component_raw",
            "active_name_key"
        ]
    ]
    .drop_duplicates()
    .head(20)
)

print(
    recovered_examples.to_string(
        index=False
    )
)

print()
print("Eloralintide normalization check:")

print(
    active_df[
        active_df["active_component_raw"]
        .astype(str)
        .str.contains(
            "Eloralintide",
            case=False,
            na=False
        )
    ][
        [
            "active_component_source",
            "active_component_raw",
            "active_name_key"
        ]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

print()
print("Top 20 normalized intervention keys by trial count:")

print(
    name_review_df[
        [
            "active_name_key",
            "trial_count",
            "example_raw_names",
            "example_sponsors"
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print()
print("=== Included trials with no retained active component ===")

if len(missing_active_trials_df) == 0:
    print("None")
else:
    columns = [
        column
        for column in [
            "nct_id",
            "brief_title",
            "conditions",
            "lead_sponsor",
            "source_intervention_types",
            "source_intervention_names"
        ]
        if column in missing_active_trials_df.columns
    ]

    print(
        missing_active_trials_df[
            columns
        ]
        .to_string(index=False)
    )

print()
print("Saved:")
print(active_file)
print(review_file)
print(missing_file)
