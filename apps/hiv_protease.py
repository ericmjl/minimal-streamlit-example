import gzip
import pickle as pkl
from io import StringIO

import holoviews as hv
import hvplot.pandas
import janitor
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from Bio import SeqIO

from pyprojroot import here

##### UTILITY FUNCTIONS #####


def predict(model, seq: str):
    """Given a sequence, predict its drug resistance value."""
    x = featurize_sequence_(seq).reshape(1, -1)
    return model.predict(x)


def predict_uncertainty(model, seq, q=[25, 75]):
    x = featurize_sequence_(seq).reshape(1, -1)
    predrange = []
    for estimator in model.estimators_:
        predrange.append(estimator.predict(x))
    minimum, maximum = np.percentile(predrange, q=q)
    return minimum, maximum


def make_preds_df(preds: np.ndarray):
    """Convenience function to make preds dataframe."""
    return (
        pd.DataFrame(preds)
        .melt()
        .rename_column("variable", "drug")
        .rename_column("value", "resistance")
    )


def make_preds_plot(preds_df: pd.DataFrame):
    """Prediction plot for drug resistance."""
    preds_plot = preds_df.hvplot.scatter(
        x="drug", y="resistance", color="red"
    ).opts(ylabel="drug resistance (higher is more resistant)")
    return preds_plot


molecular_weights = {
    "A": 89.0935,
    "R": 174.2017,
    "N": 132.1184,
    "D": 133.1032,
    "C": 121.1590,
    "E": 147.1299,
    "Q": 146.1451,
    "G": 75.0669,
    "H": 155.1552,
    "I": 131.1736,
    "L": 131.1736,
    "K": 146.1882,
    "M": 149.2124,
    "F": 165.1900,
    "P": 115.1310,
    "S": 105.0930,
    "T": 119.1197,
    "W": 204.2262,
    "Y": 181.1894,
    "V": 117.1469,
    "X": 100.00,
}


def featurize_sequence_(x, expected_size=99):
    """
    :param x: a string in a pandas DataFrame cell
    """
    feats = np.zeros(len(x))
    for i, letter in enumerate(x):
        feats[i] = molecular_weights[letter]
    return feats.reshape(1, -1)


class SequenceError(Exception):
    pass


def predict(model, sequence):
    """
    :param model: sklearn model
    :param sequence: A string, should be 99 characters long.
    """
    if len(sequence) != 99:
        raise ValueError(
            f"sequence must be of length 99. Your sequence is of length {len(sequence)}"
        )

    if not set(sequence).issubset(set(molecular_weights.keys())):
        invalid_chars = set(sequence).difference(molecular_weights.keys())
        raise SequenceError(
            f"sequence contains invalid characters: {invalid_chars}"
        )

    seqfeat = featurize_sequence_(sequence)
    return model.predict(seqfeat)


def read_fasta_file(f):
    memfile = StringIO(f.read())
    seq = SeqIO.read(memfile, format="fasta")
    return seq


##### APP BELOW #####

data_dir = here() / "data"

drugs = ["ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "SQV", "TPV"]

# Load scores to visualize.
with gzip.open(data_dir / "scores.pkl.gz", "rb") as f:
    scores = pkl.load(f)
    scores = pd.DataFrame(scores)


data = (
    pd.read_csv(here() / "data/hiv-protease-data-expanded.csv", index_col=0)
    .query("weight == 1.0")
    .transform_column("sequence", lambda x: len(x), "seq_length")
    .query("seq_length == 99")
    .transform_column("sequence", featurize_sequence_, "features")
    .transform_columns(drugs, np.log10)
)


distplot = (
    data.select_columns(drugs)
    .hvplot.box()
    .opts(xlabel="drug", ylabel="resistance", width=500, height=300)
)


protease_sequence = st.file_uploader(
    "Please upload a protease sequence.", ["fasta"]
)


if protease_sequence:
    seq = read_fasta_file(protease_sequence)
    preds = dict()

    with st.spinner("Predicting..."):
        for drug in drugs:
            with gzip.open(here() / f"data/models/{drug}.pkl.gz", "rb") as f:
                model = joblib.load(f)
            preds[drug] = predict(model, seq)
        preds_df = make_preds_df(preds).set_index("drug")

    data = (
        data.select_columns(drugs)
        .unstack()
        .reset_index()
        .rename({"level_0": "drug", 0: "resistance"}, axis="columns")
        .dropna()
    )[["drug", "resistance"]]
    drug_stats = data.groupby("drug").agg(["min", "mean", "max"])
    st.dataframe(preds_df.join(drug_stats))
