from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Next-Generation Obesity Therapeutics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        width: auto;
        max-width: none;
        box-sizing: border-box;
        padding-top: 3.2rem;
        padding-bottom: 3rem;
        padding-left: 1.6rem;
        padding-right: 1.6rem;
    }

    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden;
    }

    h1, h2, h3 {
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }

    div[data-testid="stMetricLabel"] p {
        font-size: 0.88rem;
        opacity: 0.78;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.10;
        font-weight: 760;
        letter-spacing: -0.045em;
        margin-top: 0.25rem;
        margin-bottom: 0.45rem;
        padding-top: 0.1rem;
        max-width: 980px;
    }

    .hero-subtitle {
        font-size: 0.98rem;
        opacity: 0.67;
        margin-bottom: 1.35rem;
    }

    .section-kicker {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.11em;
        opacity: 0.55;
        margin-bottom: 0.15rem;
    }

    .insight-card {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.035);
        border-radius: 14px;
        padding: 1.05rem 1.1rem;
        min-height: 138px;
    }

    .insight-card h4 {
        margin: 0 0 0.45rem 0;
        font-size: 1rem;
    }

    .insight-card .big {
        font-size: 1.65rem;
        font-weight: 720;
        line-height: 1.1;
        margin-bottom: 0.45rem;
    }

    .insight-card p {
        margin: 0;
        opacity: 0.72;
        font-size: 0.9rem;
        line-height: 1.45;
    }

    .signal-card {
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
    }

    .signal-card .title {
        font-weight: 700;
        font-size: 0.98rem;
        margin-bottom: 0.25rem;
    }

    .signal-card .meta {
        opacity: 0.72;
        font-size: 0.86rem;
        line-height: 1.4;
    }

    .method-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        background: rgba(255,255,255,0.025);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding-left: 0.65rem;
        padding-right: 0.65rem;
    }

    @media (max-width: 900px) {
        .hero-title {
            font-size: 2.3rem;
        }
    }

    /* Print / Save-as-PDF layout.
       Keeps the live dashboard full-width, but prevents browser print
       rendering from clipping the left or right edges. */
    @media print {
        html, body, [data-testid="stAppViewContainer"] {
            overflow: visible !important;
            width: 100% !important;
            max-width: 100% !important;
        }

        .block-container {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            margin: 0 !important;
            padding-left: 0.45in !important;
            padding-right: 0.45in !important;
            padding-top: 0.35in !important;
            padding-bottom: 0.35in !important;
        }

        .hero-title {
            font-size: 2.55rem !important;
            line-height: 1.12 !important;
            max-width: 100% !important;
        }

        .hero-subtitle {
            font-size: 0.9rem !important;
        }

        div[data-testid="stMetric"] {
            break-inside: avoid;
        }

        [data-testid="stPlotlyChart"] {
            max-width: 100% !important;
            overflow: visible !important;
            break-inside: avoid;
        }

        .insight-card,
        .signal-card,
        .method-card {
            break-inside: avoid;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA
# ============================================================

@st.cache_data
def load_csv(name):
    return pd.read_csv(PROCESSED / name)


trial = load_csv("phase5_trial_master.csv")
pair = load_csv("phase5_trial_therapy_master.csv")
coverage = load_csv("phase54_mechanism_coverage_by_year.csv")
strategic = load_csv("phase57_strategic_opportunity_matrix.csv")

VERIFIED_STATUSES = {"VERIFIED", "VERIFIED_ALIAS"}

SPONSOR_MAP = {
    "Novo Nordisk A/S": "Novo Nordisk",
    "Eli Lilly and Company": "Eli Lilly",
    "Rhythm Pharmaceuticals, Inc.": "Rhythm Pharmaceuticals",
    "Gan & Lee Pharmaceuticals.": "Gan & Lee Pharmaceuticals",
    "Gan and Lee Pharmaceuticals, USA": "Gan & Lee Pharmaceuticals",
    "Gasherbrum Bio, Inc., a wholly owned subsidiary of Structure Therapeutics": "Structure Therapeutics",
    "Hoffmann-La Roche": "Roche",
}


def modality_group(value):
    if pd.isna(value):
        return "Unknown / unspecified"

    s = str(value).strip().lower()

    if (
        "sirna" in s
        or "rnai" in s
        or "antisense" in s
        or "oligonucleotide" in s
    ):
        return "RNA / oligonucleotide"

    if "combination" in s or "mixed" in s:
        return "Combination / mixed modality"

    if (
        "antibody" in s
        or "fc-fusion" in s
        or "fc fusion" in s
        or "ligand trap" in s
        or "large-molecule biologic" in s
        or "multi-agonist biologic" in s
        or "multi-domain biologic" in s
        or "gdf15-fc" in s
    ):
        return "Antibody / protein biologic"

    if (
        "peptide" in s
        or "protein toxin biologic" in s
        or "cyclic dipeptide" in s
        or "cyclic peptide" in s
        or "cytokine/peptide" in s
        or "peptidomimetic" in s
    ):
        return "Peptide-based"

    if (
        "small molecule" in s
        or "small-molecule" in s
        or "aminosterol" in s
    ):
        return "Small molecule"

    if "polymer" in s or "botanical cannabinoid" in s:
        return "Gut-restricted / other"

    if "unspecified" in s or s == "":
        return "Unknown / unspecified"

    return "Gut-restricted / other"


trial["sponsor_group_final"] = (
    trial["lead_sponsor_clean"]
    .map(SPONSOR_MAP)
    .fillna(trial["lead_sponsor_clean"])
)

pair["sponsor_group_final"] = (
    pair["lead_sponsor_clean"]
    .map(SPONSOR_MAP)
    .fillna(pair["lead_sponsor_clean"])
)

pair["modality_group_final"] = pair["modality"].map(modality_group)


# ============================================================
# HELPERS
# ============================================================

def verified_only(df):
    return df[
        df["classification_status_final"].isin(VERIFIED_STATUSES)
    ].copy()


def confirmed_recent_mask(df):
    return (
        df["start_year"].between(2023, 2025)
        | (
            df["start_year"].eq(2026)
            & df["start_timing_vs_snapshot"].eq("BEFORE_SNAPSHOT")
        )
    )


def style_fig(fig, height=None):
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=42, t=45, b=28),
        legend_title_text="",
        autosize=True,
    )
    if height:
        fig.update_layout(height=height)
    return fig


def signal_card(title, meta):
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="title">{title}</div>
            <div class="meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER + COMPACT FILTER POPOVER
# ============================================================

st.markdown(
    '<div class="hero-title">Next-Generation Obesity Therapeutics</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-subtitle">
        Clinical pipeline &amp; competitive landscape · ClinicalTrials.gov ·
        Global pharmacotherapy trials, 2015–2026 · Snapshot: 17 Sep 2026
    </div>
    """,
    unsafe_allow_html=True,
)

filter_col, filter_note_col = st.columns([1, 6])

with filter_col:
    with st.popover("⚙ Filters"):
        st.caption(
            "Leave multi-selects blank to include all. "
            "Filters apply to Overview, Competition, Mechanisms and Modalities."
        )

        year_min, year_max = st.slider(
            "Trial start year",
            min_value=2015,
            max_value=2026,
            value=(2015, 2026),
            key="filter_year",
        )

        phase_options = sorted(
            trial["phase_group"].dropna().unique()
        )
        selected_phases = st.multiselect(
            "Phase",
            phase_options,
            default=[],
            placeholder="All phases",
            key="filter_phase",
        )

        sponsor_class_options = sorted(
            trial["sponsor_class_clean"].dropna().unique()
        )
        selected_sponsor_classes = st.multiselect(
            "Sponsor class",
            sponsor_class_options,
            default=[],
            placeholder="All sponsor classes",
            key="filter_sponsor_class",
        )

        mechanism_options = sorted(
            pair["mechanism_superfamily_final"]
            .dropna()
            .unique()
        )
        selected_mechanisms = st.multiselect(
            "Mechanism",
            mechanism_options,
            default=[],
            placeholder="All mechanisms",
            key="filter_mechanism",
        )

        st.caption(
            "2026 YTD counts starts definitely on/before 17 Sep 2026. "
            "Nine September month-precision records remain timing-ambiguous."
        )

with filter_note_col:
    st.caption(
        "Use Filters to narrow the analysis. Strategic Signals stays fixed to "
        "the full frozen dataset so its predefined criteria remain comparable."
    )


# ============================================================
# APPLY FILTERS
# ============================================================

trial_filtered = trial[
    trial["start_year"].between(year_min, year_max)
].copy()

if selected_phases:
    trial_filtered = trial_filtered[
        trial_filtered["phase_group"].isin(selected_phases)
    ]

if selected_sponsor_classes:
    trial_filtered = trial_filtered[
        trial_filtered["sponsor_class_clean"].isin(
            selected_sponsor_classes
        )
    ]

pair_filtered = pair[
    pair["nct_id"].isin(trial_filtered["nct_id"])
].copy()

if selected_mechanisms:
    pair_filtered = pair_filtered[
        pair_filtered["mechanism_superfamily_final"].isin(
            selected_mechanisms
        )
    ]

    allowed_ids = set(pair_filtered["nct_id"])
    trial_filtered = trial_filtered[
        trial_filtered["nct_id"].isin(allowed_ids)
    ]

verified_filtered = verified_only(pair_filtered)


# ============================================================
# KPI STRIP
# ============================================================

k1, k2, k3, k4 = st.columns(4)

k1.metric(
    "Trials",
    f"{trial_filtered['nct_id'].nunique():,}",
)

k2.metric(
    "Active",
    f"{trial_filtered.loc[trial_filtered['status_group'].eq('Active / recruiting'), 'nct_id'].nunique():,}",
)

k3.metric(
    "Industry",
    f"{trial_filtered.loc[trial_filtered['sponsor_class_clean'].eq('INDUSTRY'), 'nct_id'].nunique():,}",
)

coverage_pct = (
    len(verified_filtered) / len(pair_filtered)
    if len(pair_filtered)
    else 0
)

k4.metric(
    "Coverage",
    f"{coverage_pct:.1%}",
    help="Verified + verified-alias trial×therapy pairs in the current filter.",
)

tabs = st.tabs(
    [
        "Overview",
        "Competition",
        "Mechanisms",
        "Modalities",
        "Signals",
        "Method",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================

with tabs[0]:
    st.markdown(
        '<div class="section-kicker">Pipeline evolution</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Trial starts accelerated sharply after 2023")

    rows = []

    for year in range(year_min, year_max + 1):
        ydf = trial_filtered[
            trial_filtered["start_year"].eq(year)
        ]

        if year < 2026:
            confirmed = len(ydf)
            ambiguous = 0
        else:
            confirmed = int(
                ydf["start_timing_vs_snapshot"]
                .eq("BEFORE_SNAPSHOT")
                .sum()
            )
            ambiguous = int(
                ydf["start_timing_vs_snapshot"]
                .eq("SNAPSHOT_MONTH_MONTH_PRECISION")
                .sum()
            )

        rows.append(
            {
                "start_year": year,
                "year_label": "2026 YTD" if year == 2026 else str(year),
                "confirmed": confirmed,
                "ambiguous": ambiguous,
            }
        )

    starts = pd.DataFrame(rows)

    fig = go.Figure()
    fig.add_bar(
        x=starts["start_year"],
        y=starts["confirmed"],
        name="Confirmed starts",
        text=starts["confirmed"],
        textposition="outside",
        cliponaxis=False,
    )

    if starts["ambiguous"].sum() > 0:
        fig.add_bar(
            x=starts["start_year"],
            y=starts["ambiguous"],
            name="Sep-2026 month-only",
        )

    y_max = (
        starts["confirmed"] + starts["ambiguous"]
    ).max()

    fig.update_layout(
        barmode="stack",
        xaxis_title="Trial start year",
        yaxis_title="Trials",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        bargap=0.22,
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=starts["start_year"],
        ticktext=starts["year_label"],
        range=[year_min - 0.7, year_max + 0.9],
        tickangle=-25,
        tickfont=dict(size=10),
        automargin=True,
    )

    fig.update_yaxes(
        range=[0, y_max * 1.16]
    )

    st.plotly_chart(
        style_fig(fig, height=430),
        use_container_width=True,
    )

    st.markdown(
        '<div class="section-kicker">Executive snapshot</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Pipeline expansion</h4>
                <div class="big">17 → 206</div>
                <p>Annual starts increased from 2015 to 2025, equivalent to a 28.3% CAGR.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Mechanism concentration</h4>
                <div class="big">71.0%</div>
                <p>Incretin-based therapies represent 71.0% of verified recent mechanism activity.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            """
            <div class="insight-card">
                <h4>Next-wave biology</h4>
                <div class="big">Lean mass + RNAi</div>
                <p>Both show strong recent activity with little or no Phase 3/4 depth yet.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Headline findings above refer to the full frozen analysis population, "
        "while the chart responds to the Filters control above."
    )


# ============================================================
# COMPETITION
# ============================================================

with tabs[1]:
    st.markdown(
        '<div class="section-kicker">Competitive landscape</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Leading industry sponsors")

    industry_trials = trial_filtered[
        trial_filtered["sponsor_class_clean"].eq("INDUSTRY")
    ].copy()

    industry_pair = pair_filtered[
        pair_filtered["nct_id"].isin(industry_trials["nct_id"])
    ].copy()

    sponsor_trials = (
        industry_trials.groupby("sponsor_group_final")
        .agg(
            total_trials=("nct_id", "nunique"),
            recent_starts=(
                "nct_id",
                lambda x: x[
                    industry_trials.loc[x.index]
                    .pipe(confirmed_recent_mask)
                ].nunique()
                if len(x)
                else 0,
            ),
        )
        .reset_index()
    )

    if not sponsor_trials.empty:
        portfolio = (
            industry_pair.groupby("sponsor_group_final")
            .agg(
                unique_therapy_identities=(
                    "therapy_identity_final",
                    "nunique",
                ),
            )
            .reset_index()
        )

        verified_industry = verified_only(industry_pair)

        mechanism_breadth = (
            verified_industry.groupby("sponsor_group_final")
            .agg(
                mechanism_breadth=(
                    "mechanism_superfamily_final",
                    "nunique",
                )
            )
            .reset_index()
        )

        sponsor_df = (
            sponsor_trials
            .merge(
                portfolio,
                on="sponsor_group_final",
                how="left",
            )
            .merge(
                mechanism_breadth,
                on="sponsor_group_final",
                how="left",
            )
            .fillna(0)
        )

        sponsor_df["unique_therapy_identities"] = (
            sponsor_df["unique_therapy_identities"].astype(int)
        )
        sponsor_df["mechanism_breadth"] = (
            sponsor_df["mechanism_breadth"].astype(int)
        )

        sponsor_df = sponsor_df.sort_values(
            "total_trials",
            ascending=False,
        )

        top_n = st.slider(
            "Sponsors shown",
            min_value=5,
            max_value=min(20, max(5, len(sponsor_df))),
            value=min(12, max(5, len(sponsor_df))),
        )

        top = sponsor_df.head(top_n).sort_values(
            "total_trials"
        )

        fig = px.bar(
            top,
            x="total_trials",
            y="sponsor_group_final",
            orientation="h",
            text="total_trials",
            labels={
                "total_trials": "Unique trials",
                "sponsor_group_final": "",
            },
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_xaxes(
            range=[0, max(top["total_trials"].max() * 1.20, 1)],
            automargin=True,
        )

        st.plotly_chart(
            style_fig(fig, height=520),
            use_container_width=True,
        )

        shares = (
            sponsor_df["total_trials"]
            / sponsor_df["total_trials"].sum()
        )

        top2 = shares.head(2).sum()
        top5 = shares.head(5).sum()
        hhi = (shares.pow(2).sum()) * 10000

        m1, m2, m3 = st.columns(3)
        m1.metric("Top-2 share", f"{top2:.1%}")
        m2.metric("Top-5 share", f"{top5:.1%}")
        m3.metric("HHI", f"{hhi:.0f}")

        st.caption(
            "Lower HHI indicates a more fragmented sponsor landscape."
        )

        st.subheader("Portfolio breadth vs trial intensity")

        breadth = sponsor_df.head(30).copy()

        label_names = {
            "Novo Nordisk",
            "Eli Lilly",
            "Pfizer",
            "Amgen",
        }

        breadth["label"] = breadth["sponsor_group_final"].where(
            breadth["sponsor_group_final"].isin(label_names),
            "",
        )

        fig = px.scatter(
            breadth,
            x="unique_therapy_identities",
            y="total_trials",
            size="mechanism_breadth",
            size_max=28,
            text="label",
            hover_name="sponsor_group_final",
            hover_data=[
                "recent_starts",
                "mechanism_breadth",
            ],
            labels={
                "unique_therapy_identities": "Unique therapy identities",
                "total_trials": "Unique trials",
                "mechanism_breadth": "Mechanism breadth",
            },
        )

        fig.update_traces(
            textposition="top center",
        )

        st.plotly_chart(
            style_fig(fig, height=520),
            use_container_width=True,
        )

        st.info(
            "Trial volume and portfolio breadth are different signals: "
            "some sponsors run many trials around a smaller number of assets, "
            "while others spread activity across a broader pipeline."
        )
    else:
        st.info("No industry-sponsored trials match the current filters.")


# ============================================================
# MECHANISMS
# ============================================================

with tabs[2]:
    st.markdown(
        '<div class="section-kicker">Mechanism landscape</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Incretin-based development dominates the verified pipeline")

    if len(verified_filtered):
        mech = (
            verified_filtered.groupby("mechanism_superfamily_final")
            .agg(
                verified_pairs=("nct_id", "size"),
                unique_identities=(
                    "therapy_identity_final",
                    "nunique",
                ),
            )
            .reset_index()
        )

        mech["share"] = (
            mech["verified_pairs"] / mech["verified_pairs"].sum()
        )

        mech = mech.sort_values(
            "verified_pairs",
            ascending=False,
        )

        top_mech = mech.head(12).sort_values(
            "verified_pairs"
        )

        fig = px.bar(
            top_mech,
            x="verified_pairs",
            y="mechanism_superfamily_final",
            orientation="h",
            text=top_mech["share"].map(lambda x: f"{x:.1%}"),
            hover_data=["unique_identities"],
            labels={
                "verified_pairs": "Verified trial×therapy pairs",
                "mechanism_superfamily_final": "",
            },
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_xaxes(
            range=[0, max(top_mech["verified_pairs"].max() * 1.20, 1)],
            automargin=True,
        )

        st.plotly_chart(
            style_fig(fig, height=500),
            use_container_width=True,
        )

        st.subheader("Historical vs recent mechanism mix")

        hist = verified_filtered[
            verified_filtered["start_year"].between(2015, 2022)
        ].copy()

        recent = verified_filtered[
            confirmed_recent_mask(verified_filtered)
        ].copy()

        hist_counts = (
            hist["mechanism_superfamily_final"]
            .value_counts(normalize=True)
            .rename("2015–2022")
        )

        recent_counts = (
            recent["mechanism_superfamily_final"]
            .value_counts(normalize=True)
            .rename("2023–2026 YTD")
        )

        compare = (
            pd.concat([hist_counts, recent_counts], axis=1)
            .fillna(0)
            .rename_axis("mechanism")
            .reset_index()
        )

        top_compare_names = (
            recent["mechanism_superfamily_final"]
            .value_counts()
            .head(10)
            .index
        )

        compare = compare[
            compare["mechanism"].isin(top_compare_names)
        ].sort_values(
            "2023–2026 YTD",
            ascending=True,
        )

        fig = go.Figure()

        fig.add_bar(
            y=compare["mechanism"],
            x=compare["2015–2022"] * 100,
            name="2015–2022",
            orientation="h",
        )

        fig.add_bar(
            y=compare["mechanism"],
            x=compare["2023–2026 YTD"] * 100,
            name="2023–2026 YTD",
            orientation="h",
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="Share of verified pairs (%)",
            yaxis_title="",
        )

        st.plotly_chart(
            style_fig(fig, height=520),
            use_container_width=True,
        )

        st.info(
            "Full-dataset headline: incretin-based therapies increased from "
            "54.2% of verified pairs in 2015–2022 to 71.0% in 2023–2026 YTD, "
            "while CNS/appetite fell from 18.2% to 4.1%."
        )
    else:
        st.info("No verified mechanism pairs match the current filters.")


# ============================================================
# MODALITIES
# ============================================================

with tabs[3]:
    st.markdown(
        '<div class="section-kicker">Therapeutic modality</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Peptides remain the dominant development modality")

    if len(verified_filtered):
        mod = (
            verified_filtered.groupby("modality_group_final")
            .agg(
                verified_pairs=("nct_id", "size"),
                unique_identities=(
                    "therapy_identity_final",
                    "nunique",
                ),
            )
            .reset_index()
        )

        mod["share"] = (
            mod["verified_pairs"] / mod["verified_pairs"].sum()
        )

        mod = mod.sort_values(
            "verified_pairs",
            ascending=True,
        )

        fig = px.bar(
            mod,
            x="verified_pairs",
            y="modality_group_final",
            orientation="h",
            text=mod["share"].map(lambda x: f"{x:.1%}"),
            hover_data=["unique_identities"],
            labels={
                "verified_pairs": "Verified trial×therapy pairs",
                "modality_group_final": "",
            },
        )

        fig.update_traces(
            textposition="outside",
            cliponaxis=False,
        )

        fig.update_xaxes(
            range=[0, max(mod["verified_pairs"].max() * 1.20, 1)],
            automargin=True,
        )

        st.plotly_chart(
            style_fig(fig, height=430),
            use_container_width=True,
        )

        st.subheader("Historical vs recent modality mix")

        hist = verified_filtered[
            verified_filtered["start_year"].between(2015, 2022)
        ].copy()

        recent = verified_filtered[
            confirmed_recent_mask(verified_filtered)
        ].copy()

        hist_counts = (
            hist["modality_group_final"]
            .value_counts(normalize=True)
            .rename("2015–2022")
        )

        recent_counts = (
            recent["modality_group_final"]
            .value_counts(normalize=True)
            .rename("2023–2026 YTD")
        )

        compare = (
            pd.concat([hist_counts, recent_counts], axis=1)
            .fillna(0)
            .rename_axis("modality")
            .reset_index()
            .sort_values("2023–2026 YTD", ascending=True)
        )

        fig = go.Figure()

        fig.add_bar(
            y=compare["modality"],
            x=compare["2015–2022"] * 100,
            name="2015–2022",
            orientation="h",
        )

        fig.add_bar(
            y=compare["modality"],
            x=compare["2023–2026 YTD"] * 100,
            name="2023–2026 YTD",
            orientation="h",
        )

        fig.update_layout(
            barmode="group",
            xaxis_title="Share of verified pairs (%)",
            yaxis_title="",
        )

        st.plotly_chart(
            style_fig(fig, height=450),
            use_container_width=True,
        )

        st.info(
            "Full-dataset headline: antibody/protein biologics increased from "
            "4.2% to 8.0% of verified activity, while RNA/oligonucleotide "
            "therapies only appear in the recent period."
        )
    else:
        st.info("No verified modality data match the current filters.")


# ============================================================
# STRATEGIC SIGNALS
# ============================================================

with tabs[4]:
    st.markdown(
        '<div class="section-kicker">Strategic signals</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Emerging biology is active, but still early in development")

    st.caption(
        "This tab intentionally uses the full frozen dataset rather than sidebar "
        "filters so the predefined strategic criteria remain consistent."
    )

    plot_df = strategic[
        strategic["unique_therapy_identities"] >= 3
    ].copy()

    # Label only the large/mature reference mechanisms on-chart.
    # Emerging mechanisms are named clearly in the cards below, which avoids
    # overlapping labels around the y=0 cluster.
    label_mechanisms = {
        "Incretin-based",
        "CNS/appetite",
        "Amylin-based",
    }

    plot_df["label"] = plot_df[
        "mechanism_superfamily_final"
    ].where(
        plot_df["mechanism_superfamily_final"].isin(
            label_mechanisms
        ),
        "",
    )

    fig = px.scatter(
        plot_df,
        x="unique_therapy_identities",
        y=plot_df["phase3plus_identity_share"] * 100,
        size="recent_therapy_identities",
        size_max=34,
        color="strategic_bucket",
        text="label",
        hover_name="mechanism_superfamily_final",
        hover_data=[
            "recent_identity_share",
            "industry_sponsor_groups",
            "recent_industry_sponsor_groups",
            "strategic_bucket",
        ],
        labels={
            "unique_therapy_identities": "Unique therapy identities",
            "y": "Therapy identities reaching Phase 3/4 (%)",
            "recent_therapy_identities": "Recent identities",
        },
    )

    fig.update_traces(
        textposition="top center",
    )

    fig.update_layout(
        showlegend=False,
    )

    fig.update_xaxes(
        range=[
            -1,
            max(plot_df["unique_therapy_identities"].max() * 1.08, 5),
        ]
    )

    fig.update_yaxes(
        range=[
            -4,
            max(
                (plot_df["phase3plus_identity_share"] * 100).max() * 1.12,
                10,
            ),
        ]
    )

    st.plotly_chart(
        style_fig(fig, height=560),
        use_container_width=True,
    )

    emerging = strategic[
        strategic["emerging_early_stage_flag"].eq(True)
    ].sort_values(
        [
            "recent_therapy_identities",
            "unique_therapy_identities",
        ],
        ascending=[False, False],
    )

    established = strategic[
        strategic["established_competitive_flag"].eq(True)
    ].sort_values(
        "phase3plus_identities",
        ascending=False,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Emerging / early-stage")
        for _, row in emerging.iterrows():
            signal_card(
                row["mechanism_superfamily_final"],
                (
                    f"{int(row['unique_therapy_identities'])} identities · "
                    f"{row['recent_identity_share']:.0%} recent · "
                    f"{int(row['phase3plus_identities'])} Phase 3/4 · "
                    f"{int(row['industry_sponsor_groups'])} industry sponsors"
                ),
            )

    with right:
        st.markdown("#### Established / competitive")
        for _, row in established.iterrows():
            signal_card(
                row["mechanism_superfamily_final"],
                (
                    f"{int(row['unique_therapy_identities'])} identities · "
                    f"{row['phase3plus_identity_share']:.0%} reached Phase 3/4 · "
                    f"{int(row['industry_sponsor_groups'])} industry sponsors"
                ),
            )

    st.warning(
        "These are descriptive strategic signals, not commercial forecasts. "
        "‘Lower sponsor density’ means fewer sponsor groups per therapy identity, "
        "not necessarily low absolute competition."
    )


# ============================================================
# METHODOLOGY
# ============================================================

with tabs[5]:
    st.markdown(
        '<div class="section-kicker">Methodology</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Frozen analysis population")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Included trials", "986")
    m2.metric("Trial×therapy pairs", "1,186")
    m3.metric("Therapy identities", "371")
    m4.metric("Verified pairs", "972", "82.0%")

    st.subheader("Mechanism-classification coverage")

    fig = px.line(
        coverage,
        x="start_year",
        y=coverage["verification_coverage"] * 100,
        markers=True,
        labels={
            "start_year": "Trial start year",
            "y": "Verification coverage (%)",
        },
    )

    fig.update_yaxes(range=[70, 90])

    st.plotly_chart(
        style_fig(fig, height=380),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="method-card">
                <strong>Scope</strong><br><br>
                Global ClinicalTrials.gov interventional studies<br>
                Obesity/overweight pharmacotherapy development<br>
                Trial starts from 2015–2026<br>
                Conventional pharmacological treatments only
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="method-card">
                <strong>Classification status</strong><br><br>
                972 verified mechanism pairs (82.0%)<br>
                26 researched but publicly unresolved (2.2%)<br>
                188 unreviewed long-tail pairs (15.9%)<br>
                0 identity-mapping conflicts
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Detailed exclusions and 2026 YTD handling"):
        st.markdown(
            """
            **Excluded from scope**
            - Nutraceuticals and traditional/herbal interventions
            - Non-pharmacological procedures
            - Drugs studied only because participants had obesity rather than to treat obesity
            - Supportive medications unrelated to obesity treatment

            **2026 YTD**
            - Snapshot date: 17 Sep 2026
            - 200 trials definitely started by the snapshot
            - 9 September 2026 month-precision records have uncertain exact timing
            - 17 future planned starts are excluded from YTD counts

            **Mechanism charts**
            - Shares use verified / verified-alias pairs only
            - Unresolved and long-tail assets remain in the underlying dataset but are excluded from mechanism-share calculations
            """
        )

st.divider()
st.caption(
    "Frozen ClinicalTrials.gov snapshot · Trial-level and trial×therapy analyses "
    "are kept separate to avoid double-counting multi-focal studies."
)
