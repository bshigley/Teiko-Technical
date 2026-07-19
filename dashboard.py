#!/usr/bin/env python3
"""Interactive dashboard for the cell-count analysis.

Run with:  streamlit run dashboard.py   (or `make dashboard`)

Reads the SQLite database built by `load_data.py`. All queries are reused from
analysis.py so the dashboard and the batch pipeline can never drift apart.
"""

import os
import sqlite3

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    AVG_BCELL_SQL,
    BASELINE_CTE,
    COHORT_SQL,
    DB_PATH,
    POPULATION_ORDER,
    _frequency_level_tests,
    summary_table,
)

# Responder vs non-responder colours: two well-separated categorical hues
# (validated CVD-safe blue / orange from the data-viz reference palette).
RESPONSE_COLORS = {"yes": "#2a78d6", "no": "#eb6834"}
RESPONSE_LABELS = {"yes": "responder", "no": "non-responder"}

st.set_page_config(page_title="Loblaw Bio - cell-count dashboard", layout="wide")


@st.cache_resource
def get_connection():
    # On a fresh host (e.g. Streamlit Community Cloud) the .db is git-ignored and
    # absent, so build it from the bundled CSV on first run. Locally it already
    # exists from `make pipeline` and this is skipped.
    if not os.path.exists(DB_PATH):
        try:
            import load_data
            load_data.load()
        except Exception:
            return None
    # check_same_thread=False: Streamlit reruns can touch the conn from threads.
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_data
def load_summary():
    return summary_table(get_connection())


@st.cache_data
def load_cohort():
    return pd.read_sql_query(COHORT_SQL, get_connection())


conn = get_connection()
st.title("Bob Loblaw Bio Inc")
st.caption(
    "Analysis for Bob Loblaw's miraclib trial. Built from "
    "`cell-count.db` (run `python load_data.py` first)."
)

if conn is None:
    st.error(
        "Database `cell-count.db` not found. Build it first with "
        "`python load_data.py`, then reload this page."
    )
    st.stop()

tab2, tab3, tab4 = st.tabs(
    ["Overview", "Responders vs non-responders",
     "Baseline subset"]
)

with tab2:
    st.header("Relative frequency of each cell population per sample")
    st.write(
        "For each sample, `count` is a population's cell count and `percentage` "
        "is that count as a share of the sample's total across all five "
        "populations."
    )
    summary = load_summary()

    c1, c2 = st.columns(2)
    c1.metric("Samples", f"{summary['sample'].nunique():,}")
    c2.metric("Rows (samples x populations)", f"{len(summary):,}")

    f1, f2 = st.columns([2, 3])
    pops = f1.multiselect(
        "Populations", POPULATION_ORDER, default=POPULATION_ORDER
    )
    query = f2.text_input("Filter by sample id (substring)", "")

    view = summary[summary["population"].isin(pops)]
    if query:
        view = view[view["sample"].str.contains(query, case=False, na=False)]

    st.dataframe(view, use_container_width=True, hide_index=True, height=430)
    st.download_button(
        "Download this table (CSV)",
        view.to_csv(index=False).encode(),
        file_name="summary_cell_frequencies.csv",
        mime="text/csv",
    )

with tab3:
    st.header("Relative frequencies: miraclib responders vs non-responders")
    st.write(
        "Cohort: **melanoma** patients on **miraclib**, **PBMC** samples only, "
        "with a known yes/no response."
    )
    cohort = load_cohort()
    n_resp = cohort.loc[cohort["response"] == "yes", "subject"].nunique()
    n_non = cohort.loc[cohort["response"] == "no", "subject"].nunique()
    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", cohort["subject"].nunique())
    c2.metric("Responders", n_resp)
    c3.metric("Non-responders", n_non)

    plot_df = cohort.copy()
    plot_df["Response"] = plot_df["response"].map(RESPONSE_LABELS)
    fig = px.box(
        plot_df,
        x="population",
        y="percentage",
        color="response",
        category_orders={"population": POPULATION_ORDER, "response": ["yes", "no"]},
        color_discrete_map=RESPONSE_COLORS,
        labels={"population": "Immune cell population",
                "percentage": "Relative frequency (% of sample total)",
                "response": "Response"},
        points="outliers",
    )
    fig.for_each_trace(
        lambda t: t.update(name=RESPONSE_LABELS.get(t.name, t.name))
    )
    fig.update_layout(legend_title_text="Response", boxmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which populations differ significantly?")
    st.write(
        "Each subject is collapsed to its **mean relative frequency across days "
        "0/7/14** (one independent observation per subject), then a Mann-Whitney "
        "U test compares responders vs non-responders per population, with a "
        "Benjamini-Hochberg FDR correction across the five populations."
    )
    results = _frequency_level_tests(cohort)
    st.dataframe(results, use_container_width=True, hide_index=True)

    sig = results.loc[results["significant (FDR<0.05)"], "population"].tolist()
    if sig:
        st.success(
            f"Significant after FDR correction: {', '.join(sig)}."
        )
    else:
        raw = results.loc[results["significant (p<0.05)"], "population"].tolist()
        note = (f" ({', '.join(raw)} is suggestive at raw p<0.05 but does not "
                f"survive correction)") if raw else ""
        st.info(
            "No population shows a statistically robust difference in relative "
            f"frequency between responders and non-responders{note}. See the "
            "README for the supplementary trajectory analyses."
        )

with tab4:
    st.header("Baseline subset: melanoma PBMC, miraclib, time = 0")

    totals = pd.read_sql_query(
        BASELINE_CTE + "SELECT COUNT(*) AS samples, "
        "COUNT(DISTINCT subject_id) AS subjects FROM baseline", conn)
    per_project = pd.read_sql_query(
        BASELINE_CTE + "SELECT project, COUNT(*) AS samples FROM baseline "
        "GROUP BY project ORDER BY project", conn)
    by_response = pd.read_sql_query(
        BASELINE_CTE + "SELECT COALESCE(response,'unknown') AS response, "
        "COUNT(DISTINCT subject_id) AS subjects FROM baseline "
        "GROUP BY COALESCE(response,'unknown') ORDER BY response", conn)
    by_sex = pd.read_sql_query(
        BASELINE_CTE + "SELECT sex, COUNT(DISTINCT subject_id) AS subjects "
        "FROM baseline GROUP BY sex ORDER BY sex", conn)

    c1, c2 = st.columns(2)
    c1.metric("Samples", int(totals.loc[0, "samples"]))
    c2.metric("Subjects", int(totals.loc[0, "subjects"]))

    g1, g2, g3 = st.columns(3)
    with g1:
        st.caption("Samples per project")
        st.plotly_chart(
            px.bar(per_project, x="project", y="samples",
                   color_discrete_sequence=["#2a78d6"]),
            use_container_width=True)
    with g2:
        st.caption("Subjects by response")
        st.plotly_chart(
            px.bar(by_response, x="response", y="subjects",
                   color="response", color_discrete_map=RESPONSE_COLORS),
            use_container_width=True)
    with g3:
        st.caption("Subjects by sex")
        st.plotly_chart(
            px.bar(by_sex, x="sex", y="subjects",
                   color_discrete_sequence=["#4a3aa7"]),
            use_container_width=True)

    st.subheader("Average B cells - melanoma males, responders, time = 0")
    st.caption("All sample types and all treatments (only condition, sex, "
               "response, and timepoint are constrained).")
    row = pd.read_sql_query(AVG_BCELL_SQL, conn).iloc[0]
    a1, a2 = st.columns(2)
    a1.metric("Average B-cell count", f"{float(row['avg_b_cells']):.2f}")
    a2.metric("Samples", int(row["n_samples"]))
