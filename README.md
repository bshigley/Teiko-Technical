# Bob Loblaw Bio Technical for Teiko (Love this by the way)

Loads `cell-count.csv` into a normalized SQLite database and answers Bob
Loblaw's questions about immune-cell population frequencies and miraclib
response (Parts 1–4), plus an interactive dashboard.

As specified in the form, the `Makefile` is what drives the process:

```bash
make setup      # install dependencies from requirements.txt
make pipeline   # do all of the analysis and draw charts with pretty colors
make dashboard  # start the dashboard
```

- **`make setup`** installs `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`,
  `statsmodels`, `scikit-learn`, `streamlit`, and `plotly`.
- **`make pipeline`** runs `load_data.py` then `analysis.py`, with no manual
  steps, again as requested. It creates `cell-count.db`, prints every result, and writes the output
  files listed below.
- **`make dashboard`** launches Streamlit headless on port 8501. In Codespaces a
  "port forwarded" toast appears — open it to view the dashboard. (Override the
  port with `make dashboard PORT=8600`.)

### Outputs produced by `make pipeline`

| File | Contents |
|------|----------|
| `cell-count.db` | SQLite database (Part 1) |
| `outputs/summary_cell_frequencies.csv` | Part 2 relative-frequency table |
| `responders_vs_nonresponders_boxplot.png` | Part 3 boxplot |
| `responders_vs_nonresponders_longitudinal.png` | Part 3 supplementary trajectory plot |

## Dashboard

`make dashboard` serves an interactive Streamlit app with three tabs — Part 2
(filterable summary table + CSV download), Part 3 (responder-vs-non-responder
boxplot and significance table), and Part 4 (baseline breakdowns and the average
B-cell figure). It reads `cell-count.db`, so run `make pipeline` first.

**Live link:** _<do not forget to update dumbass>_ — to publish a public link, push
this repo to GitHub and deploy `dashboard.py` on
[Streamlit Community Cloud](https://share.streamlit.io) (free; point it at
`dashboard.py`). The app also runs locally / in Codespaces via `make dashboard`.

`load_data.py` takes no arguments and creates `cell-count.db` in the repo root.
`analysis.py` reads from that database and runs every analysis in order.

## Schema

The two main ideas are that we want to store the subject level info that doesn't change in only one place, and then be a bit craftier with the way the cell counts are stored,
as described below.

| Table         | Grain                            | Columns |
|---------------|----------------------------------|---------|
| `subjects`    | one row per patient              | subject_id (PK), project, condition, age, sex, treatment, response |
| `samples`     | one row per physical sample      | sample_id (PK), subject_id (FK), sample_type, time_from_treatment_start |
| `cell_counts` | one row per (sample, population) | sample_id (FK), population, count — PK (sample_id, population) |

`cell_counts` is stored in **long format** with one row per cell type per sample. Very importantly, this means that you can use simple GROUP BY statements to simplify the queries.
And equally importantly, if you wanted to add more populations, you simply expand the data rather than changing the entire schema. sujbect_id and sample_id are the join keys here.

### How this scales

- **Keep updates small and consistent.** Mr. Loblaw would not approve of computationally expensive and risky update routines that have to change a million
things in a million different places. In our case, the subject metadata lives in
one place, so a corrected response value is a single-row update. You do NOT want a bunch of things in one place, especially if that happens to be US currency inside a banana stand.

- **The cell count format allows us to expand to new populations easily**
You basically only have to add new rows here and you can continue grouping by population, easy money.

- **Joins shouldn't be too expensive**
Most of the actual computation happens within the cell counts table, and the joins are not expensive with this schema.

- **Further analysis** This is standard SQL so no matter how many there are, it can be easily migrated to PostgreSQL where you can do fancier stuff in parallel as the number of projects
expands.

## Code structure

- **`load_data.py`** Exactly what your spec asked for and with the schema defined above.
- **`analysis.py`** All of the actual analysis! Of course, putting everything in a single file would make neither Mr. Bob Loblaw nor the engineers at Teiko especially happy in production,
but I did it this way for your ease of grading / reading and tried to decompose clearly. 
- **`dashboard.py`** — the Streamlit UI. It **imports the same query constants
  and analysis functions from `analysis.py`**, so the dashboard and the batch
  pipeline can never disagree on a number.

## Data overview

`analysis.py` prints one row per (sample, population) with `sample`,
`total_count`, `population`, `count`, and `percentage` — the population count as
a percentage of that sample's total across all five populations. The percentage
is computed in SQL, and the full table is also written to
`outputs/summary_cell_frequencies.csv`.

## Part 3 — Responders vs non-responders

Cohort: melanoma patients on miraclib, **PBMC samples only**, with a known
`yes`/`no` response (656 subjects, 1,968 samples across days 0/7/14).

### Primary analysis

You asked which populations show a significant difference in relative
**frequencies** between responders and non-responders. 

The first important thing is to avoid pseudoreplication since there are 3 samples per subject, which are guaranteed to be highly correlated and thus will break any independence assumption.
And so I collapsed it to its **mean relative frequency across all 3 timepoints** and then I did the **Mann-Whitney U test** because biological data is rarely normally distributed. I then did
a Benjamini-Hochberg FDR correction across each of the 5 populations. The required **boxplot** (`responders_vs_nonresponders_boxplot.png`) shows the per-population distributions split by response.

**Result:** `cd4_t_cell` is suggestive (raw *p* ≈ 0.012) but **does not survive
FDR correction** (adjusted *p* ≈ 0.06), while every other population is clearly
non-significant. **No immune population shows a statistically robust difference in
relative frequency between miraclib responders and non-responders in this
cohort.**

### Supplementary analyses (truth be told, I initially read the spec to ask for this but am now choosing to parlay my findings into additional analysis)

Since Bob Loblaw is principally concerned with *predicting* response, `analysis.py` also runs some fun things that take into account the changes between 0, 7, and 14 days.

1. **Per-subject interval deltas** — Mann-Whitney U on each subject's change in
   frequency over day 0→7 and 7→14 (one delta per subject per interval), FDR
   across the ten tests. Nothing is significant.
2. **Linear mixed-effects model** — per population, a likelihood-ratio test for
   the `time × response` interaction in
   `percentage ~ C(timepoint) * C(response)` with a per-subject random intercept.
   `b_cell` reaches raw *p* ≈ 0.04 but does **not** survive FDR (adjusted
   ≈ 0.20).
3. **Random forest** — one row per subject (baseline CLR composition + change in
   CLR over each interval), stratified 5-fold CV. Out-of-fold ROC-AUC ≈ 0.53,
   barely above chance.

All three agree with the primary test that the spec asked for: B cells are the only population that ever
hints at a signal, and that hint never survives multiple-testing correction. **The
honest conclusion for Yah: these five population frequencies do not provide a
statistically robust predictor of miraclib response in this cohort.**

## Part 4 — Data subset analysis

### Baseline subset

Melanoma PBMC samples at baseline (`time_from_treatment_start = 0`) from
miraclib-treated patients — **656 samples / 656 subjects**:

- **Samples per project:** prj1 = 384, prj3 = 272. (prj2 is absent because its
  melanoma+miraclib samples are all whole-blood, not PBMC — a real data feature.)
- **Subjects by response:** responders (yes) = 331, non-responders (no) = 325.
- **Subjects by sex:** female = 312, male = 344.

### Average B cells — melanoma males, responders, baseline

Average B-cell count for melanoma **male** responders at
`time_from_treatment_start = 0`, across **all** sample types and treatments (only
condition, sex, response, and timepoint constrained): **10206.15** over 485
samples.