#!/usr/bin/env python3
import os
import sqlite3
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "cell-count.csv")
DB_PATH = os.path.join(BASE_DIR, "cell-count.db")

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"] #The 5 cell types in the csv in the order on which they appear

#The most important choice here is to have the cell_counts table have 5 rows per sample, one row per cell type so that you can easily group by in order to calculate the freqs asked for
SCHEMA = """
DROP TABLE IF EXISTS cell_counts;
DROP TABLE IF EXISTS samples;
DROP TABLE IF EXISTS subjects;

CREATE TABLE subjects (
    subject_id  TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    condition   TEXT NOT NULL,
    age         INTEGER,
    sex         TEXT,
    treatment   TEXT,
    response    TEXT
);

CREATE TABLE samples (
    sample_id                  TEXT PRIMARY KEY,
    subject_id                 TEXT NOT NULL REFERENCES subjects(subject_id),
    sample_type                TEXT NOT NULL,
    time_from_treatment_start  INTEGER
);

CREATE TABLE cell_counts (
    sample_id   TEXT NOT NULL REFERENCES samples(sample_id),
    population  TEXT NOT NULL,              -- one of POPULATIONS
    count       INTEGER NOT NULL,
    PRIMARY KEY (sample_id, population)
);

CREATE INDEX idx_samples_subject   ON samples(subject_id);
CREATE INDEX idx_cellcounts_sample ON cell_counts(sample_id);
"""


def load(csv_path=CSV_PATH, db_path=DB_PATH):
    df = pd.read_csv(csv_path)
    df = df.astype(object).where(pd.notnull(df), None)

    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)

        subjects = (
            df[["subject", "project", "condition", "age", "sex", "treatment", "response"]]
            .drop_duplicates(subset="subject"))

        connection.executemany(
            "INSERT INTO subjects "
            "(subject_id, project, condition, age, sex, treatment, response) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            subjects.itertuples(index=False, name=None))

        samples = df[["sample", "subject", "sample_type", "time_from_treatment_start"]]
        connection.executemany(
            "INSERT INTO samples "
            "(sample_id, subject_id, sample_type, time_from_treatment_start) "
            "VALUES (?, ?, ?, ?)",
            samples.itertuples(index=False, name=None))

        long = df.melt(
            id_vars=["sample"],
            value_vars=POPULATIONS,
            var_name="population",
            value_name="count")
        
        connection.executemany(
            "INSERT INTO cell_counts (sample_id, population, count) VALUES (?, ?, ?)",
            long[["sample", "population", "count"]].itertuples(index=False, name=None))

        connection.commit()

    finally:
        connection.close()

if __name__ == "__main__":
    load()
