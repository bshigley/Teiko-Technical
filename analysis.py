#!/usr/bin/env python3
import os
import sqlite3
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cell-count.db")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

POPULATION_ORDER = ["b_cell", "cd4_t_cell", "cd8_t_cell", "monocyte", "nk_cell"]
ALPHA = 0.05
RF_SEED = 42

def connect(db_path=DB_PATH):
    if not os.path.exists(db_path):
        raise SystemExit(
            f"{db_path} not found. Run `python load_data.py` first to build it."
        )
    return sqlite3.connect(db_path)


# Find the relative percentages of each cell type
SUMMARY_SQL = """
SELECT
    cc.sample_id                                    AS sample,
    totals.total_count                              AS total_count,
    cc.population                                   AS population,
    cc.count                                        AS count,
    ROUND(100.0 * cc.count / totals.total_count, 4) AS percentage
FROM cell_counts AS cc
JOIN (
    SELECT sample_id, SUM(count) AS total_count
    FROM cell_counts
    GROUP BY sample_id
) AS totals ON totals.sample_id = cc.sample_id
ORDER BY cc.sample_id, cc.population
"""


def summary_table(connection):
    return pd.read_sql_query(SUMMARY_SQL, connection)

def summary(connection, csv_path=None):
    table = summary_table(connection)

    display = table.copy()
    display["percentage"] = display["percentage"].round(2)

    print("=" * 70)
    print("Part 2 - Relative frequency of each cell population per sample")
    print("=" * 70)
    with pd.option_context("display.max_rows", 30, "display.width", 120):
        print(display.to_string(index=False))
    print(f"\n{len(table)} rows  ({table['sample'].nunique()} samples x "
          f"{table['population'].nunique()} populations)")

    if csv_path:
        table.to_csv(csv_path, index=False)
        print(f"Summary table written to {csv_path}")
    return table

# Cohort filter + per-sample frequency in one query: melanoma + miraclib + PBMC,
# restricted to subjects with a known yes/no response.
COHORT_SQL = """
WITH totals AS (
    SELECT sample_id, SUM(count) AS total_count
    FROM cell_counts
    GROUP BY sample_id
)
SELECT
    s.sample_id                        AS sample,
    sub.subject_id                     AS subject,
    sub.response                       AS response,
    cc.population                      AS population,
    s.time_from_treatment_start        AS timepoint,
    cc.count                           AS count,
    t.total_count                      AS total_count,
    100.0 * cc.count / t.total_count   AS percentage
FROM samples     AS s
JOIN subjects    AS sub ON sub.subject_id = s.subject_id
JOIN cell_counts AS cc  ON cc.sample_id  = s.sample_id
JOIN totals      AS t   ON t.sample_id   = s.sample_id
WHERE sub.condition = 'melanoma'
  AND sub.treatment = 'miraclib'
  AND s.sample_type = 'PBMC'
  AND sub.response IN ('yes', 'no')
ORDER BY cc.population, sub.response, sub.subject_id, s.time_from_treatment_start
"""

def _responder_tests(df):
    """Compare each population's change over treatment, responders vs non responders.

    Avoid pseudoreplication and get rid of the Mann-Whitney U independence assumption.
    Instead look at deltas from 0 -> 7 and 7 -> 14 and then do the FDR rigamarole (not Franlin Delano Roosevelt as my brain insists on saying no matter what).
    """
    # One row per (subject, response, population); timepoints become columns.
    wide = df.pivot(
        index=["subject", "response", "population"],
        columns="timepoint", values="percentage",
    ).reset_index()

    id_cols = ["subject", "response", "population"]
    times = sorted(c for c in wide.columns if c not in id_cols)
    intervals = list(zip(times[:-1], times[1:]))  # consecutive pairs, e.g. (0,7),(7,14)

    rows = []
    for pop in POPULATION_ORDER:
        sub = wide[wide["population"] == pop]
        for lo, hi in intervals:
            delta = sub[hi] - sub[lo]  # per-subject change over this interval
            responders = delta[sub["response"] == "yes"].dropna().to_numpy()
            non = delta[sub["response"] == "no"].dropna().to_numpy()

            # Important : no normality assumption, suits skewed frequencies which biological samples mostly are.
            u_stat, p = stats.mannwhitneyu(responders, non, alternative="two-sided")

            rows.append({
                "population": pop,
                "interval": f"day {lo}->{hi}",
                "n_responders": len(responders),
                "n_non_responders": len(non),
                "median_responder_delta": round(np.median(responders), 3),
                "median_non_responder_delta": round(np.median(non), 3),
                "u_statistic": u_stat,
                "p_value": p,
            })

    res = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # Benjamini-Hochberg FDR across every (population, interval) test
    res["p_adjusted"] = stats.false_discovery_control(res["p_value"], method="bh")

    res["significant (p<0.05)"] = res["p_value"] < ALPHA
    res["significant (FDR<0.05)"] = res["p_adjusted"] < ALPHA
    return res


def _responder_boxplot(df, path):
    """Boxplot of relative frequency per population, responders vs non-responders.

    The required Part 3 visualization: for each immune-cell population, the
    distribution of PBMC relative frequencies split by response. This is a
    descriptive view of the frequencies themselves; the significance claim comes
    from _frequency_level_tests, which collapses each subject to one mean
    frequency to handle the repeated-measures structure that a pooled boxplot
    does not.
    """
    sns.set_theme(style="whitegrid")
    palette = {"yes": "#2a9d8f", "no": "#e76f51"}

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(
        data=df, x="population", y="percentage", hue="response",
        order=POPULATION_ORDER, hue_order=["yes", "no"], palette=palette,
        fliersize=0, ax=ax,
    )
    sns.stripplot(
        data=df, x="population", y="percentage", hue="response",
        order=POPULATION_ORDER, hue_order=["yes", "no"], dodge=True,
        palette="dark:.25", size=2.5, alpha=0.4, ax=ax, legend=False,
    )
    ax.set_title(
        "Immune-cell relative frequencies: miraclib responders vs non-responders\n"
        "(melanoma, PBMC samples)"
    )
    ax.set_xlabel("Immune cell population")
    ax.set_ylabel("Relative frequency (% of sample total)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ["responder (yes)", "non-responder (no)"], title="Response")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _responder_longitudinal_plot(df, path):
    """Longitudinal trajectory of each population, responders vs non-responders.

    A boxplot of pooled samples hides how the populations drift over treatment;
    a per-population line plot over time shows the trend the delta test targets.
    """
    sns.set_theme(style="whitegrid")
    palette = {"yes": "#2a9d8f", "no": "#e76f51"}

    g = sns.relplot(
        data=df, x="timepoint", y="percentage", hue="response",
        col="population", col_order=POPULATION_ORDER, kind="line",
        palette=palette, hue_order=["yes", "no"],
        errorbar=("ci", 95), marker="o", height=4, aspect=1.2,
    )
    g.fig.suptitle(
        "Longitudinal immune-cell trajectories: miraclib responders vs "
        "non-responders (melanoma, PBMC)", y=1.05,
    )
    g.set_axis_labels("Days from treatment start", "Relative frequency (%)")
    g.set_titles("{col_name}")
    if g.legend is not None:
        g.legend.set_title("Response")
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)

def _frequency_level_tests(df):
    """This is the main question the spec asked, which is whether there is a significant difference in the frequencies between responders and non-responders.
    We want to do Mann-Whitney U still because of the non-normality assumption, but to avoid pseudoreplication we need to collapse each subject to a single value, which I defined as just
    the average relative frequency across the 3 time points. Then we run the test and do the FDR correction.
    """
    per_subject = (
        df.groupby(["subject", "response", "population"])["percentage"]
        .mean()
        .reset_index()
    )

    rows = []
    for pop in POPULATION_ORDER:
        sub = per_subject[per_subject["population"] == pop]
        responders = sub.loc[sub["response"] == "yes", "percentage"].to_numpy()
        non = sub.loc[sub["response"] == "no", "percentage"].to_numpy()
        u_stat, p = stats.mannwhitneyu(responders, non, alternative="two-sided")
        rows.append({
            "population": pop,
            "n_responders": len(responders),
            "n_non_responders": len(non),
            "median_responder_pct": round(float(np.median(responders)), 3),
            "median_non_responder_pct": round(float(np.median(non)), 3),
            "u_statistic": u_stat,
            "p_value": p,
        })

    res = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    res["p_adjusted"] = stats.false_discovery_control(res["p_value"], method="bh")
    res["significant (p<0.05)"] = res["p_value"] < ALPHA
    res["significant (FDR<0.05)"] = res["p_adjusted"] < ALPHA
    return res


def frequency_comparison(connection, box_path):
    """Primary Part 3 deliverable: compare relative-frequency levels + boxplot."""
    df = pd.read_sql_query(COHORT_SQL, connection)
    if df.empty:
        raise SystemExit(
            "No qualifying samples (melanoma + miraclib + PBMC with yes/no response)."
        )

    print("\n" + "$" * 69)
    print("Responders vs non-responders: relative-frequency levels "
          "(melanoma + miraclib + PBMC)")
    print("$" * 69)
    n_resp = df.loc[df["response"] == "yes", "subject"].nunique()
    n_non = df.loc[df["response"] == "no", "subject"].nunique()
    print(f"Cohort: {df['subject'].nunique()} subjects "
          f"(responders: {n_resp}, non-responders: {n_non}); each subject "
          f"collapsed to its mean relative frequency across days 0/7/14, so "
          f"1 subject = 1 data point.\n")

    results = _frequency_level_tests(df)
    with pd.option_context("display.width", 180, "display.max_columns", None):
        print(results.to_string(index=False))

    _responder_boxplot(df, box_path)
    print(f"\nBoxplot written to {box_path}")

    sig = results.loc[results["significant (p<0.05)"], "population"].tolist()
    if sig:
        print(f"\nSignificant difference in relative frequency at raw p < {ALPHA}: "
              f"{', '.join(sig)}.")
        sig_fdr = results.loc[results["significant (FDR<0.05)"], "population"].tolist()
        print("Still significant after Benjamini-Hochberg FDR correction: "
              f"{', '.join(sig_fdr) if sig_fdr else 'none'}.")
    else:
        print(f"\nNo population's relative frequency differs significantly between "
              f"responders and non-responders at p < {ALPHA}.")
    return results

def responders(connection, line_path):
    df = pd.read_sql_query(COHORT_SQL, connection)
    if df.empty:
        raise SystemExit(
            "No qualifying samples (melanoma + miraclib + PBMC with yes/no response)."
        )

    print("\n" + "$" * 69)
    print("Responders vs non-responders: change over "
          "treatment (melanoma + miraclib + PBMC)")
    print("$" * 69)
    n_resp = df.loc[df["response"] == "yes", "subject"].nunique()
    n_non = df.loc[df["response"] == "no", "subject"].nunique()
    times = sorted(int(t) for t in df["timepoint"].unique())
    print(f"Cohort: {df['subject'].nunique()} subjects "
          f"(responders: {n_resp}, non-responders: {n_non}); one delta per "
          f"subject for each consecutive interval across days {times}\n")

    results = _responder_tests(df)
    with pd.option_context("display.width", 180, "display.max_columns", None):
        print(results.to_string(index=False))

    _responder_longitudinal_plot(df, line_path)
    print(f"\nLongitudinal plot written to {line_path}")

    def _label(rows):
        return ", ".join(f"{r.population} ({r.interval})" for r in rows.itertuples())

    sig = results[results["significant (p<0.05)"]]
    if not sig.empty:
        print(f"\nSignificant change at raw p < {ALPHA} "
              f"(Mann-Whitney U on interval deltas): {_label(sig)}.")
        sig_fdr = results[results["significant (FDR<0.05)"]]
        print("Still significant after Benjamini-Hochberg FDR correction: "
              f"{_label(sig_fdr) if not sig_fdr.empty else 'none'}.")
    else:
        print(f"\nNo population's change over any interval differs significantly "
              f"at p < {ALPHA}.")
    return results

def _mixed_effects_tests(df):
    """Per population, test the time x response interaction in a mixed model.

    For each population we fit

        percentage ~ C(timepoint) * C(response),  random intercept per subject

    This is an additional sanity check.
    """
    rows = []
    for pop in POPULATION_ORDER:
        sub = df.loc[
            df["population"] == pop, ["subject", "response", "timepoint", "percentage"]
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence benign convergence chatter
            full = smf.mixedlm(
                "percentage ~ C(timepoint) * C(response)", sub, groups=sub["subject"]
            ).fit(reml=False, method="lbfgs")
            reduced = smf.mixedlm(
                "percentage ~ C(timepoint) + C(response)", sub, groups=sub["subject"]
            ).fit(reml=False, method="lbfgs")

        # LRT: the interaction adds (n_timepoints - 1) fixed-effect terms.
        df_diff = len(full.fe_params) - len(reduced.fe_params)
        lr_stat = max(2.0 * (full.llf - reduced.llf), 0.0)
        p = stats.chi2.sf(lr_stat, df_diff)

        rows.append({
            "population": pop,
            "lr_chi2": round(lr_stat, 3),
            "df": df_diff,
            "p_value": p,
        })

    res = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    res["p_adjusted"] = stats.false_discovery_control(res["p_value"], method="bh")
    res["significant (p<0.05)"] = res["p_value"] < ALPHA
    res["significant (FDR<0.05)"] = res["p_adjusted"] < ALPHA
    return res


def mixed_model(connection):
    df = pd.read_sql_query(COHORT_SQL, connection)
    if df.empty:
        raise SystemExit("No qualifying samples for the mixed-effects model.")

    print("\n" + "$" * 69)
    print("Mixed-effects model: time x response interaction "
          "(all timepoints)")
    print("$" * 69)
    print("Per population: LRT for the C(timepoint) x C(response) interaction in "
          "percentage ~ C(timepoint) * C(response) with a per-subject random "
          "intercept\n")

    results = _mixed_effects_tests(df)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(results.to_string(index=False))

    sig = results.loc[results["significant (p<0.05)"], "population"].tolist()
    if sig:
        print(f"\nSignificant time x response interaction at p < {ALPHA}: "
              f"{', '.join(sig)}.")
        sig_fdr = results.loc[results["significant (FDR<0.05)"], "population"].tolist()
        print("Still significant after FDR: "
              f"{', '.join(sig_fdr) if sig_fdr else 'none'}.")
    else:
        print(f"\nNo significant difference in trajectory with p < {ALPHA}.")
    return results

def _clr(counts):
    """We need to do a centered log-ratio transform here. Since the frequencies are mathematically bounded to equal 100, they are not independent so we need to adjust to
    a space where the computer can walk in the forest or whatever. 
    """
    log_counts = np.log(counts.astype(float))
    return log_counts - log_counts.mean(axis=1, keepdims=True)


def _subject_features(df):
    """Here we basically want to figure out how the frequencies are changing across timepoints, e.g. 0 -> 7 and from 7 -> 14. Important for the random forest and we don't want to
    only look from 0 -> 14.
    """
    times = sorted(int(t) for t in df["timepoint"].unique())

    def clr_frame(t):
        counts = (
            df[df["timepoint"] == t]
            .pivot(index=["subject", "response"], columns="population", values="count")
            [POPULATION_ORDER]
        )
        return pd.DataFrame(
            _clr(counts.to_numpy()), index=counts.index, columns=POPULATION_ORDER
        )

    clr_by_time = {t: clr_frame(t) for t in times}
    base = clr_by_time[times[0]].sort_index()

    blocks = [base.rename(columns={p: f"clr_base_{p}" for p in POPULATION_ORDER})]
    for lo, hi in zip(times[:-1], times[1:]):
        delta = clr_by_time[hi].reindex(base.index) - clr_by_time[lo].reindex(base.index)
        blocks.append(
            delta.rename(columns={p: f"clr_delta_{lo}_{hi}_{p}" for p in POPULATION_ORDER})
        )
    feats = pd.concat(blocks, axis=1)

    y = (feats.index.get_level_values("response") == "yes").astype(int)
    return feats, np.asarray(y), times


def random_forest(connection):
    df = pd.read_sql_query(COHORT_SQL, connection)
    if df.empty:
        raise SystemExit("No samples for the random forest.")

    feats, y, times = _subject_features(df)
    X = feats.to_numpy()
    feature_names = list(feats.columns)
    intervals = ", ".join(f"{lo}->{hi}" for lo, hi in zip(times[:-1], times[1:]))

    print("\n" + "$" * 69)
    print("Random forest: predicting miraclib response "
          "(time-aware, per-subject)")
    print("$" * 69)
    print(f"{len(feats)} subjects, one row each; features = baseline CLR (day "
          f"{times[0]}) + change in CLR over each interval (days {intervals}), "
          f"per population ({len(feature_names)} features)")

    clf = RandomForestClassifier(
        n_estimators=500, random_state=RF_SEED, class_weight="balanced", n_jobs=-1,
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RF_SEED)

    proba = cross_val_predict(
        clf, X, y, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y, proba)
    print(f"\nStratifiedKFold(5) out-of-fold ROC-AUC: {auc:.3f}  "
          "(0.5 = no better than chance)")
    print("\nClassification report (out-of-fold):")
    print(classification_report(y, pred, target_names=["non-responder", "responder"]))

    # Permutation importance, as one of my old professors would have preferred
    fold_importances = []
    for train_idx, test_idx in cv.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        result = permutation_importance(
            clf, X[test_idx], y[test_idx], scoring="roc_auc",
            n_repeats=25, random_state=RF_SEED, n_jobs=-1,
        )
        fold_importances.append(result.importances_mean)

    importances = (
        pd.Series(np.mean(fold_importances, axis=0), index=feature_names)
        .sort_values(ascending=False)
    )
    print("Permutation importance :")
    print(importances.round(4).to_string())

    if auc < 0.6:
        print("\nWeak sauce unfortunately, difficult to distinguish from pure chance")

    return auc, importances

# All sample types and all treatments where only the condition, sex, response, time,
# and population are constrained.
AVG_BCELL_SQL = """
SELECT AVG(cc.count) AS avg_b_cells, COUNT(*) AS n_samples
FROM cell_counts AS cc
JOIN samples     AS s   ON s.sample_id  = cc.sample_id
JOIN subjects    AS sub ON sub.subject_id = s.subject_id
WHERE cc.population = 'b_cell'
  AND sub.condition = 'melanoma'
  AND sub.sex = 'M'
  AND sub.response = 'yes'
  AND s.time_from_treatment_start = 0
"""


def avg_bcell_melanoma_male_responders(connection):
    row = pd.read_sql_query(AVG_BCELL_SQL, connection).iloc[0]

    print("\n" + "$" * 69)
    print("Part 4 - Average B cells: melanoma males, responders, time=0 "
          "(all sample & treatment types)")
    print("$" * 69)
    n = int(row["n_samples"])
    if n == 0:
        print("No matching samples.")
        return None
    avg = float(row["avg_b_cells"])
    print(f"Samples: {n}")
    print(f"Average B-cell count: {avg:.2f}")
    return avg


# Melanoma + miraclib + PBMC + baseline (time_from_treatment_start = 0).
BASELINE_CTE = """
WITH baseline AS (
    SELECT s.sample_id, s.subject_id, sub.project, sub.response, sub.sex
    FROM samples  AS s
    JOIN subjects AS sub ON sub.subject_id = s.subject_id
    WHERE sub.condition = 'melanoma'
      AND sub.treatment = 'miraclib'
      AND s.sample_type = 'PBMC'
      AND s.time_from_treatment_start = 0
)
"""

def baseline(connection):
    totals = pd.read_sql_query(
        BASELINE_CTE + "SELECT COUNT(*) AS samples, "
        "COUNT(DISTINCT subject_id) AS subjects FROM baseline", connection)
    per_project = pd.read_sql_query(
        BASELINE_CTE + "SELECT project, COUNT(*) AS samples FROM baseline "
        "GROUP BY project ORDER BY project", connection)
    by_response = pd.read_sql_query(
        BASELINE_CTE + "SELECT COALESCE(response, 'unknown') AS response, "
        "COUNT(DISTINCT subject_id) AS subjects FROM baseline "
        "GROUP BY COALESCE(response, 'unknown') ORDER BY response", connection)
    by_sex = pd.read_sql_query(
        BASELINE_CTE + "SELECT sex, COUNT(DISTINCT subject_id) AS subjects "
        "FROM baseline GROUP BY sex ORDER BY sex", connection)

    print("\n" + "$" * 69)
    print("Baseline subset "
          "(melanoma PBMC, miraclib, time_from_treatment_start = 0)")
    print("$" * 69)
    print(f"{int(totals.loc[0, 'samples'])} samples from "
          f"{int(totals.loc[0, 'subjects'])} subjects\n")
    print("Samples per project:")
    print(per_project.to_string(index=False))
    print("\nSubjects by response:")
    print(by_response.to_string(index=False))
    print("\nSubjects by sex:")
    print(by_sex.to_string(index=False))


BOXPLOT_PATH = os.path.join(BASE_DIR, "responders_vs_nonresponders_boxplot.png")
LINEPLOT_PATH = os.path.join(BASE_DIR, "responders_vs_nonresponders_longitudinal.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_csv = os.path.join(OUTPUT_DIR, "summary_cell_frequencies.csv")

    with connect() as connection:
        summary(connection, csv_path=summary_csv)
        
        frequency_comparison(connection, box_path=BOXPLOT_PATH)
        responders(connection, line_path=LINEPLOT_PATH)
        mixed_model(connection)
        random_forest(connection)
        
        avg_bcell_melanoma_male_responders(connection)
        baseline(connection)


if __name__ == "__main__":
    main()