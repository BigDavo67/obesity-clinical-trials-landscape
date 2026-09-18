from pathlib import Path
import re
import pandas as pd


# ============================================================
# PHASE 4.6 — PREPARE CANONICAL THERAPY + ROLE REVIEW
#
# Purpose:
#   1. Map verified brands/development codes to a canonical name.
#   2. Flag non-focal pharmacological components such as DDI probes.
#   3. Export review tables for the remaining uncertain names/roles.
#
# IMPORTANT:
#   This script creates CANDIDATE classifications only.
#   It does not silently guess unknown development codes.
# ============================================================

processed_data_dir = Path("data/processed")

stage1_file = (
    processed_data_dir
    / "obesity_active_interventions_stage1.csv"
)

trials_file = (
    processed_data_dir
    / "obesity_trials_clean.csv"
)

name_review_file = (
    processed_data_dir
    / "intervention_name_review.csv"
)


active_df = pd.read_csv(stage1_file)
trials_df = pd.read_csv(trials_file)
name_review_df = pd.read_csv(name_review_file)

print("=== Phase 4.6: canonical therapy preparation ===")
print("Active intervention rows:", len(active_df))
print("Included trials:", len(trials_df))
print("Basic normalized keys:", len(name_review_df))


# ============================================================
# 1. VERIFIED ALIAS MAP
# ============================================================

# Exact aliases verified from ClinicalTrials.gov / PubMed / current
# regulatory or publication records.
#
# Store names in lowercase because active_name_key is lowercase.

exact_alias_map = {

    # --------------------------------------------------------
    # Novo Nordisk
    # --------------------------------------------------------
    "nnc0174-0833": "cagrilintide",

    # NNC0487-0111 was initially called amycretin.
    # Current 2026 records use the active-substance name zenagamtide.
    "nnc0487-0111": "zenagamtide",
    "amycretin": "zenagamtide",

    # --------------------------------------------------------
    # Boehringer Ingelheim
    # --------------------------------------------------------
    "bi 456906": "survodutide",
    "bi456906": "survodutide",

    # --------------------------------------------------------
    # Innovent / Lilly
    # --------------------------------------------------------
    "ibi362": "mazdutide",
    "ly3305677": "mazdutide",

    # --------------------------------------------------------
    # Hanmi
    # --------------------------------------------------------
    "hm15211": "efocipegtrutide",

    # --------------------------------------------------------
    # Sciwind
    # --------------------------------------------------------
    "xw003": "ecnoglutide",

    # --------------------------------------------------------
    # AstraZeneca / MedImmune
    # --------------------------------------------------------
    "medi0382": "cotadutide",
    "medi-0382": "cotadutide",

    # --------------------------------------------------------
    # Pfizer
    # --------------------------------------------------------
    "pf-07081532": "lotiglipron",
    "pf07081532": "lotiglipron",

    "pf-06882961": "danuglipron",
    "pf06882961": "danuglipron",

    # --------------------------------------------------------
    # Lilly
    # --------------------------------------------------------
    "ly3502970": "orforglipron",
    "ly-3502970": "orforglipron",

    "ly3437943": "retatrutide",
    "ly-3437943": "retatrutide",

    # --------------------------------------------------------
    # Common brands
    # --------------------------------------------------------
    "wegovy": "semaglutide",
    "ozempic": "semaglutide",
    "rybelsus": "semaglutide",

    "saxenda": "liraglutide",

    "mounjaro": "tirzepatide",
    "zepbound": "tirzepatide",

    "belviq": "lorcaserin",

    "xenical": "orlistat",
    "alli": "orlistat",

    "contrave": "naltrexone + bupropion",

    "qsymia": "phentermine + topiramate",

    # --------------------------------------------------------
    # High-impact 2026 verified aliases / current names
    # --------------------------------------------------------
    "hrs9531": "ribupatide",
    "hrs-9531": "ribupatide",
    "kai-9531": "ribupatide",

    "gzr18": "bofanglutide",

    "hdm1005": "poterepatide",

    "ly3841136": "eloralintide",
    "ly-3841136": "eloralintide",

    "ly3549492": "naperiglipron",
    "ly-3549492": "naperiglipron",

    "pf-08653944": "berobenatide",
    "pf08653944": "berobenatide",
    "met097": "berobenatide",
    "met-097": "berobenatide",
    "met-097i": "berobenatide",

    "amg 133": "maridebart cafraglutide",
    "amg133": "maridebart cafraglutide",

    "lik066": "licogliflozin",

    "gsbr-1290": "aleniglipron",
    "gsbr1290": "aleniglipron",

    # MWN109 appears in arm labels as "MWN109 1/2/3".
    # These are formulation/dose-arm labels, not different molecules.
    "mwn109 1": "mwn109",
    "mwn109 2": "mwn109",
    "mwn109 3": "mwn109",
}


# ============================================================
# 2. SOURCE URLS FOR VERIFIED ALIASES
# ============================================================

alias_source_map = {
    "nnc0174-0833":
        "https://clinicaltrials.gov/study/NCT05564104",

    "nnc0487-0111":
        "https://ctis.eu/trial/2026-525209-11-00",

    "amycretin":
        "https://ctis.eu/trial/2026-525209-11-00",

    "bi 456906":
        "https://pubchem.ncbi.nlm.nih.gov/compound/Survodutide",

    "bi456906":
        "https://pubchem.ncbi.nlm.nih.gov/compound/Survodutide",

    "ibi362":
        "https://pubmed.ncbi.nlm.nih.gov/36247927/",

    "ly3305677":
        "https://pubmed.ncbi.nlm.nih.gov/36247927/",

    "hm15211":
        "https://pubmed.ncbi.nlm.nih.gov/37028504/",

    "xw003":
        "https://pubmed.ncbi.nlm.nih.gov/37364710/",

    "medi0382":
        "https://pubchem.ncbi.nlm.nih.gov/compound/Cotadutide",

    "medi-0382":
        "https://pubchem.ncbi.nlm.nih.gov/compound/Cotadutide",

    "pf-07081532":
        "https://pubmed.ncbi.nlm.nih.gov/38751362/",

    "pf07081532":
        "https://pubmed.ncbi.nlm.nih.gov/38751362/",

    "pf-06882961":
        "https://pubmed.ncbi.nlm.nih.gov/40539310/",

    "pf06882961":
        "https://pubmed.ncbi.nlm.nih.gov/40539310/",

    "ly3502970":
        "https://pubmed.ncbi.nlm.nih.gov/37344954/",

    "ly-3502970":
        "https://pubmed.ncbi.nlm.nih.gov/37344954/",

    "ly3437943":
        "https://clinicaltrials.gov/search?intr=LY3437943",

    "ly-3437943":
        "https://clinicaltrials.gov/search?intr=LY3437943",

    "hrs9531":
        "https://www.hengrui.com/en/media/detail-991.html",
    "hrs-9531":
        "https://www.hengrui.com/en/media/detail-991.html",
    "kai-9531":
        "https://www.kailera.com/ribupatide/",

    "gzr18":
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC13130691/",

    "hdm1005":
        "https://pubs.acs.org/doi/10.1021/acs.jmedchem.6c00466",

    "ly3841136":
        "https://pubmed.ncbi.nlm.nih.gov/41109426/",
    "ly-3841136":
        "https://pubmed.ncbi.nlm.nih.gov/41109426/",

    "ly3549492":
        "https://www.lilly.com/science/research-development/pipeline",
    "ly-3549492":
        "https://www.lilly.com/science/research-development/pipeline",

    "pf-08653944":
        "https://clinicaltrials.gov/study/NCT07595549",
    "pf08653944":
        "https://clinicaltrials.gov/study/NCT07595549",
    "met097":
        "https://www.pfizer.com/news/press-release/press-release-detail/pfizers-ultra-long-acting-injectable-glp-1-ra-shows-robust",
    "met-097":
        "https://www.pfizer.com/news/press-release/press-release-detail/pfizers-ultra-long-acting-injectable-glp-1-ra-shows-robust",
    "met-097i":
        "https://www.pfizer.com/news/press-release/press-release-detail/pfizers-ultra-long-acting-injectable-glp-1-ra-shows-robust",

    "amg 133":
        "https://clinicaltrials.gov/search?intr=AMG%20133",
    "amg133":
        "https://clinicaltrials.gov/search?intr=AMG%20133",

    "lik066":
        "https://pubmed.ncbi.nlm.nih.gov/32187881/",

    "gsbr-1290":
        "https://pubchem.ncbi.nlm.nih.gov/compound/Aleniglipron",
    "gsbr1290":
        "https://pubchem.ncbi.nlm.nih.gov/compound/Aleniglipron",

    "mwn109 1":
        "https://diabetesjournals.org/diabetes/article/74/Supplement_1/1967-LB/158767/",
    "mwn109 2":
        "https://diabetesjournals.org/diabetes/article/74/Supplement_1/1967-LB/158767/",
    "mwn109 3":
        "https://diabetesjournals.org/diabetes/article/74/Supplement_1/1967-LB/158767/",
}


# ============================================================
# 3. SAFE CANONICALIZATION RULES
# ============================================================

def clean_key(value):
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

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


def canonicalize_name(key):
    """
    Return:
        canonical_candidate
        mapping_status
        mapping_rule
        source_url

    Unknown coded drugs remain unchanged and are flagged for research.
    """
    key = clean_key(key)

    if not key:
        return "", "EMPTY", "", ""

    # --------------------------------------------------------
    # 3A. Exact verified alias
    # --------------------------------------------------------

    if key in exact_alias_map:
        return (
            exact_alias_map[key],
            "VERIFIED_ALIAS",
            f"exact alias: {key}",
            alias_source_map.get(key, "")
        )

    # --------------------------------------------------------
    # 3B. Verified aliases with formulation/cohort suffixes
    # --------------------------------------------------------

    prefix_aliases = [
        ("nnc0487-0111", "zenagamtide"),
        ("ibi362", "mazdutide"),
        ("pf-07081532", "lotiglipron"),
        ("pf-06882961", "danuglipron"),
    ]

    for prefix, canonical in prefix_aliases:
        if key.startswith(prefix + " "):
            return (
                canonical,
                "VERIFIED_ALIAS",
                f"verified code prefix: {prefix}",
                alias_source_map.get(prefix, "")
            )

    # --------------------------------------------------------
    # 3C. Brand/generic wording variants
    # --------------------------------------------------------

    if "wegovy" in key or "ozempic" in key or "rybelsus" in key:
        return (
            "semaglutide",
            "BRAND_GENERIC",
            "semaglutide brand name",
            ""
        )

    if "saxenda" in key:
        return (
            "liraglutide",
            "BRAND_GENERIC",
            "liraglutide brand name",
            ""
        )

    if "mounjaro" in key or "zepbound" in key:
        return (
            "tirzepatide",
            "BRAND_GENERIC",
            "tirzepatide brand name",
            ""
        )

    if "belviq" in key:
        return (
            "lorcaserin",
            "BRAND_GENERIC",
            "lorcaserin brand name",
            ""
        )

    # Xenical/Alli strings may include comparison wording.
    if "xenical" in key or key == "alli":
        return (
            "orlistat",
            "BRAND_GENERIC",
            "orlistat brand name",
            ""
        )

    # --------------------------------------------------------
    # 3D. CagriSema combination
    # --------------------------------------------------------

    if "cagrisema" in key:
        return (
            "cagrilintide + semaglutide",
            "COMBINATION_STANDARDIZED",
            "CagriSema",
            "https://clinicaltrials.gov/search?intr=CagriSema"
        )

    # --------------------------------------------------------
    # 3E. Naltrexone/bupropion / Contrave
    # --------------------------------------------------------

    naltrexone_bupropion_signal = (
        "naltrexone" in key
        and "bupropion" in key
    )

    if naltrexone_bupropion_signal or "contrave" in key:
        # Preserve metformin when explicitly part of a triple regimen.
        if "metformin" in key:
            return (
                "naltrexone + bupropion + metformin",
                "COMBINATION_STANDARDIZED",
                "multi-drug regimen",
                ""
            )

        return (
            "naltrexone + bupropion",
            "COMBINATION_STANDARDIZED",
            "Contrave / NB combination",
            ""
        )

    # --------------------------------------------------------
    # 3F. Phentermine/topiramate / Qsymia
    # --------------------------------------------------------

    phen_top_signal = (
        "phentermine" in key
        and "topiramate" in key
    )

    if phen_top_signal or "qsymia" in key:
        return (
            "phentermine + topiramate",
            "COMBINATION_STANDARDIZED",
            "Qsymia / phentermine-topiramate",
            ""
        )

    # --------------------------------------------------------
    # 3G. Verified development codes with no public generic
    #     name that we need to substitute.
    #
    # These are valid therapy identities, not unresolved aliases.
    # --------------------------------------------------------

    verified_code_as_canonical = {
        "bgm0504",
        "azd6234",
        "ubt251",
        "hrs-7535",
        "hsg4112",
        "da-302168s",
        "nnc0662-0419",
        "mdr-001",
        "mwn109",
    }

    if key in verified_code_as_canonical:
        return (
            key,
            "VERIFIED_CODE_AS_CANONICAL",
            "verified development code; no alias substitution required",
            ""
        )

    # Also collapse residual MWN109 arm-label variants.
    if key.startswith("mwn109 "):
        return (
            "mwn109",
            "VERIFIED_CODE_AS_CANONICAL",
            "MWN109 arm/formulation label collapsed to molecule code",
            "https://diabetesjournals.org/diabetes/article/74/Supplement_1/1967-LB/158767/"
        )

    # UBT251 normalization can retain '(ID ...)' remnants as 'ubt251 id'.
    if key.startswith("ubt251"):
        return (
            "ubt251",
            "VERIFIED_CODE_AS_CANONICAL",
            "UBT251 arm-label variant collapsed to molecule code",
            "https://www.novonordisk.com/news-and-media/news-and-ir-materials/news-details.html?id=916503"
        )

    # --------------------------------------------------------
    # 3H. Common generic names already suitable as canonical
    # --------------------------------------------------------

    generic_names = {
        "semaglutide",
        "tirzepatide",
        "liraglutide",
        "cagrilintide",
        "orforglipron",
        "maridebart cafraglutide",
        "eloralintide",
        "metformin",
        "setmelanotide",
        "retatrutide",
        "phentermine",
        "orlistat",
        "danuglipron",
        "lotiglipron",
        "survodutide",
        "mazdutide",
        "ecnoglutide",
        "cotadutide",
        "efocipegtrutide",
        "zenagamtide",
        "lorcaserin",
        "topiramate",
        "bupropion",
        "naltrexone",
        "dapagliflozin",
        "exenatide",
        "dulaglutide",
        "pramlintide",
        "tesofensine",

        # Current generic/nonproprietary names
        "ribupatide",
        "bofanglutide",
        "poterepatide",
        "naperiglipron",
        "berobenatide",
        "licogliflozin",
        "aleniglipron",
        "bimagrumab",
        "lisdexamfetamine dimesylate",

        # Common PK/DDI/background medicines: canonical as written
        "acetaminophen",
        "paracetamol",
        "digoxin",
        "midazolam",
        "warfarin",
        "rosuvastatin",
        "atorvastatin",
        "moxifloxacin",
        "caffeine",
        "omeprazole",
    }

    if key in generic_names:
        return (
            key,
            "CANONICAL_AS_IS",
            "already generic/canonical",
            ""
        )

    # --------------------------------------------------------
    # 3I. Likely development code: preserve, do not guess
    # --------------------------------------------------------

    code_patterns = [
        r"^[a-z]{1,6}-?\d{2,}[a-z0-9-]*$",
        r"^[a-z]{1,6}\s?\d{2,}[a-z0-9 -]*$",
        r"^nnc\d",
        r"^ly\d",
        r"^pf-\d",
        r"^ibi\d",
        r"^hrs[-]?\d",
        r"^hm\d",
        r"^azd\d",
        r"^bgm\d",
        r"^hdm\d",
        r"^ubt\d",
        r"^mwn\d",
        r"^jnj-\d",
    ]

    if any(
        re.search(pattern, key)
        for pattern in code_patterns
    ):
        return (
            key,
            "UNMAPPED_CODE",
            "development code requires targeted verification",
            ""
        )

    # --------------------------------------------------------
    # 3J. Everything else
    # --------------------------------------------------------

    return (
        key,
        "UNMAPPED_TEXT",
        "requires review",
        ""
    )


canonical_results = (
    name_review_df["active_name_key"]
    .apply(canonicalize_name)
)

name_review_df[
    [
        "canonical_candidate",
        "mapping_status",
        "mapping_rule",
        "mapping_source_url"
    ]
] = pd.DataFrame(
    canonical_results.tolist(),
    index=name_review_df.index
)


# ============================================================
# 4. ROLE HINTS — NAME LEVEL ONLY
# ============================================================

# These are not final trial-specific roles.
# They simply help prioritize obvious non-focal drugs.

likely_pk_ddi_probe_names = {
    "acetaminophen",
    "paracetamol",
    "digoxin",
    "midazolam",
    "warfarin",
    "rosuvastatin",
    "atorvastatin",
    "moxifloxacin",
    "caffeine",
    "omeprazole",
    "dextromethorphan",
}

likely_background_names = {
    "contraceptive",
    "oral contraceptive",
}

def assign_role_hint(row):

    key = clean_key(row["active_name_key"])
    canonical = clean_key(
        row["canonical_candidate"]
    )

    if (
        key in likely_pk_ddi_probe_names
        or canonical in likely_pk_ddi_probe_names
    ):
        return "LIKELY_PK_DDI_PROBE"

    if (
        key in likely_background_names
        or canonical in likely_background_names
    ):
        return "LIKELY_BACKGROUND_OR_DDI"

    # Focal obesity drugs still need trial-level checking because the
    # same medicine can sometimes be a comparator or background therapy.
    if row["mapping_status"] in {
        "VERIFIED_ALIAS",
        "BRAND_GENERIC",
        "COMBINATION_STANDARDIZED",
        "CANONICAL_AS_IS",
        "VERIFIED_CODE_AS_CANONICAL",
    }:
        return "POTENTIAL_FOCAL_OR_COMPARATOR"

    return "ROLE_REVIEW_REQUIRED"


name_review_df["role_hint"] = (
    name_review_df.apply(
        assign_role_hint,
        axis=1
    )
)


# ============================================================
# 5. ADD TRIAL CONTEXT TO ACTIVE INTERVENTION ROWS
# ============================================================

context_columns = [
    "nct_id",
    "brief_title",
    "official_title",
    "conditions",
    "phase",
    "primary_purpose",
    "overall_status",
    "lead_sponsor",
    "lead_sponsor_class",
    "brief_summary",
    "primary_outcome_measures",
    "primary_outcome_descriptions",
]

available_context_columns = [
    column
    for column in context_columns
    if column in trials_df.columns
]

trial_context_df = (
    trials_df[
        available_context_columns
    ]
    .drop_duplicates(subset=["nct_id"])
)

trial_intervention_df = active_df.merge(
    trial_context_df,
    on="nct_id",
    how="left",
    suffixes=("", "_trial")
)

# Merge the name-level canonical candidate and role hint.
name_mapping_columns = [
    "active_name_key",
    "canonical_candidate",
    "mapping_status",
    "mapping_rule",
    "mapping_source_url",
    "role_hint",
]

trial_intervention_df = (
    trial_intervention_df
    .merge(
        name_review_df[
            name_mapping_columns
        ],
        on="active_name_key",
        how="left"
    )
)


# ============================================================
# 6. TRIAL-SPECIFIC ROLE SCREENING
# ============================================================

def combine_context(row):

    columns = [
        "brief_title",
        "official_title",
        "conditions",
        "brief_summary",
        "primary_outcome_measures",
        "primary_outcome_descriptions",
    ]

    values = []

    for column in columns:
        value = row.get(column, "")

        if pd.notna(value):
            values.append(
                str(value)
            )

    return " ".join(values).lower()


trial_intervention_df["role_review_text"] = (
    trial_intervention_df.apply(
        combine_context,
        axis=1
    )
)


ddi_context_phrases = [
    "drug-drug interaction",
    "drug drug interaction",
    "ddi",
    "cytochrome",
    "cyp3a",
    "cyp2c",
    "cyp2d6",
    "pharmacokinetic interaction",
    "cocktail",
    "probe substrate",
]

qt_context_phrases = [
    "qtc",
    "qt interval",
    "thorough qt",
]

background_context_phrases = [
    "background therapy",
    "background medication",
    "concomitant",
]

def trial_role_candidate(row):

    role_hint = row["role_hint"]
    text = row["role_review_text"]

    if role_hint == "LIKELY_PK_DDI_PROBE":
        return "PK_DDI_PROBE"

    if any(
        phrase in text
        for phrase in ddi_context_phrases
    ):
        # Do not automatically label the investigational obesity drug
        # itself as a probe just because the trial is a DDI study.
        if clean_key(
            row["canonical_candidate"]
        ) in likely_pk_ddi_probe_names:
            return "PK_DDI_PROBE"

    if any(
        phrase in text
        for phrase in qt_context_phrases
    ):
        if clean_key(
            row["canonical_candidate"]
        ) == "moxifloxacin":
            return "POSITIVE_CONTROL"

    if role_hint == "LIKELY_BACKGROUND_OR_DDI":
        return "BACKGROUND_OR_CONCOMITANT"

    if any(
        phrase in text
        for phrase in background_context_phrases
    ):
        return "CONTEXT_REVIEW"

    return "FOCAL_VS_COMPARATOR_REVIEW"


trial_intervention_df[
    "trial_role_candidate"
] = (
    trial_intervention_df.apply(
        trial_role_candidate,
        axis=1
    )
)


# ============================================================
# 7. CREATE PRIORITIZED NAME REVIEW
# ============================================================

# High-frequency unknown names should be researched first.
name_review_df["needs_alias_research"] = (
    name_review_df["mapping_status"]
    .isin(
        [
            "UNMAPPED_CODE",
            "UNMAPPED_TEXT"
        ]
    )
)

name_review_df = (
    name_review_df
    .sort_values(
        by=[
            "needs_alias_research",
            "trial_count",
            "active_name_key"
        ],
        ascending=[
            False,
            False,
            True
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 8. SAVE OUTPUTS
# ============================================================

name_output_file = (
    processed_data_dir
    / "phase46_name_mapping_review.csv"
)

trial_role_output_file = (
    processed_data_dir
    / "phase46_trial_intervention_review.csv"
)

name_review_df.to_csv(
    name_output_file,
    index=False
)

trial_intervention_df.to_csv(
    trial_role_output_file,
    index=False
)


# ============================================================
# 9. VALIDATION / SUMMARY
# ============================================================

print()
print("=== Canonical-name mapping status ===")
print(
    name_review_df["mapping_status"]
    .value_counts(dropna=False)
)

print()
print("=== Name-level role hints ===")
print(
    name_review_df["role_hint"]
    .value_counts(dropna=False)
)

print()
print("=== Trial-intervention role candidates ===")
print(
    trial_intervention_df[
        "trial_role_candidate"
    ]
    .value_counts(dropna=False)
)

print()
print(
    "Unique canonical candidates:",
    name_review_df[
        "canonical_candidate"
    ].nunique()
)

print(
    "Names still requiring alias research:",
    name_review_df[
        "needs_alias_research"
    ].sum()
)

print()
print("--- Top 30 unresolved names by trial count ---")

unresolved = (
    name_review_df[
        name_review_df[
            "needs_alias_research"
        ]
    ][
        [
            "active_name_key",
            "trial_count",
            "example_raw_names",
            "example_sponsors",
            "mapping_status"
        ]
    ]
    .head(30)
)

print(
    unresolved.to_string(index=False)
)

print()
print("--- Likely PK/DDI probes ---")

probe_rows = (
    name_review_df[
        name_review_df[
            "role_hint"
        ].isin(
            [
                "LIKELY_PK_DDI_PROBE",
                "LIKELY_BACKGROUND_OR_DDI"
            ]
        )
    ][
        [
            "active_name_key",
            "canonical_candidate",
            "trial_count",
            "role_hint",
            "example_sponsors"
        ]
    ]
)

print(
    probe_rows.to_string(index=False)
)

print()
print("Saved:")
print(name_output_file)
print(trial_role_output_file)
