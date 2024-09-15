#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kaniuth
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from utils.utils import correlate_matrices


def evaluate_prediction_from(method, results, true_rsms):
    """Evaluate a method's predicted RMs against ground-truth RSMs.

    For each method, every predicted RM of any given DNN-module for any
    imageset is Pearson correlated with its respective ground-truth RM.

    Parameters
    ----------
    method : str
        Denotes the results of which method shall be used.
    results : pd.DataFrame
        Holds the results as obtained with `method`.
    true_rsms : dict
        Holds the ground-truth RMs for each imageset.

    Returns
    -------
    df : pd.DataFrame
        Holds the Pearson correlation between predicted and ground-truth
        RMs together with various information pertaining to that
        specific case. Columns are as follows:

        =============   ===================================================================
        imgset          The imageset for which embedding was predicted (as `str`)
        model_module    The specific DNN-module architectur (as `str`)
        regularization  Regularization scheme used when predicting (as `str`)
        f'{method}_r'   Correlation for `method` between predicted and true RM (as `float`)
        =============   ===================================================================
    """
    assert method in ["dimpred", "crsa"], '\nWrong arg for "method"\n'
    df = pd.DataFrame(columns=[])
    for imgset in results["imgset"].unique():
        true_rsm = true_rsms[imgset]
        cond1 = results["imgset"] == imgset
        for model_module in results.loc[cond1, "model_module"].unique():
            cond2 = cond1 & (results["model_module"] == model_module)
            if "kriegeskorte" in imgset:
                matrix_kind = "rdm_euclidean"
            else:
                if method == "dimpred":
                    matrix_kind = "rsm_spose"
                elif method == "crsa":
                    matrix_kind = "rsm_pearson"
            predicted_rsm = results.loc[cond2, matrix_kind].tolist()[0]
            r, _ = correlate_matrices(true_rsm, predicted_rsm)
            df_i = pd.DataFrame(columns=[])
            df_i.loc[0, "imgset"] = imgset
            df_i.loc[0, "model_module"] = model_module
            df_i.loc[0, f"{method}_r"] = r
            df = pd.concat([df, df_i], ignore_index=True)
    df.sort_values(by=["imgset", f"{method}_r"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def select_dnns_per_dim(df, n):
    """Select the best n DNNs for every condition.

    Imagine several DNNs predicted a ground-truth SPoSE embedding for an
    imageset. Now one could, for each SPoSE dimension, select those `n`
    DNNs, that were most predictive. This is achieved by this function.

    Parameters
    ----------
    df : pd.DataFrame
        Holds the correlation between the DNN-predicted embedding and
        the ground-truth embedding. Expected shape is (ndim,
        DNN).
    n : int
        Indicates the n best DNNs that shall be determined for each
        dimension.

    Returns
    -------
    df2 : pd.DataFrame
        Holds the best n DNNs for each dimension. Shape is (ndim, Top `n`),
        and the values are the DNN names.
    """
    assert n <= len(df.axes[1]), '\n"n" is bigger than there are cols in "df"\n'
    columns = [f"Top ({i+1})" for i in range(n)]
    df2 = pd.DataFrame(
        df.apply(lambda x: x.nlargest(n).index.tolist(), axis=1).tolist(),
        columns=columns,
    )
    return df2


def create_plot_order_for(row, col, vals):
    """Create new order column in a pd.df."""
    return vals.index(row[col])


def correlate_embeddings(embedding1, embedding2):
    """Correlates two embeddings with each other.

    Each dimension (i.e. column) of the first embedding is correlated
    with its counterpart of the second embedding.

    Parameters
    ----------
    embedding1 : ndarray
        First embedding. Usually the embedding as predicted by
        `model_module`. Expected shape is (1854, ndims). ndims will be
        either 49 or 66.
    embedding2 : ndarray
        Second embedding. Usually the true SPoSE embedding. Expected
        shape is (1854, ndims). ndims will be either 49 or 66.
        The shape must match that of `embedding1`.

    Returns
    -------
    embedding_corrs : ndarray
        The Pearson correlations for all dimensions of the embedding.
        Shape is (ndim).
    """
    n_dim = embedding2.shape[1]
    embedding_corrs = np.zeros((n_dim))
    for dim in range(n_dim):
        x = embedding2[:, dim]
        y = embedding1[:, dim]
        r, _ = pearsonr(x, y)
        embedding_corrs[dim] = r
    return embedding_corrs


def predict_triplet_behavior(actual_triplet, predicted_rsm):
    """Calculate whether a behavioral triplet choice can be predicted.

    For an actual triplet choice, calculate whether the representational matrix
    can correctly predict the choice. This func is applied to a pd.df's rows.

    Parameters
    ----------
    actual_triplet : pd.df row
        The pd.df. row contains three columns, one for each presented image
        coded as an integer.
    predicted_rsm : np.ndarray
        Holds a representational matrix with (dis-)similarity values.

    Returns
    -------
    int
        Whether the predicted choice equals the actual choice (1) or not (0).
    """
    i1_i2_similarity = predicted_rsm[
        actual_triplet.image1 - 1, actual_triplet.image2 - 1
    ]  # the "-1" is necessary since the columns in "actual_triplet" have Matlab indices.
    i1_i3_similarity = predicted_rsm[
        actual_triplet.image1 - 1, actual_triplet.image3 - 1
    ]
    i2_i3_similarity = predicted_rsm[
        actual_triplet.image2 - 1, actual_triplet.image3 - 1
    ]
    all_similarities = [i1_i2_similarity, i1_i3_similarity, i2_i3_similarity]
    max_simimilarity_pair = all_similarities.index(max(all_similarities))
    if max_simimilarity_pair == 2:
        predicted_choice = 1
    elif max_simimilarity_pair == 1:
        predicted_choice = 2
    elif max_simimilarity_pair == 0:
        predicted_choice = 3
    if predicted_choice == actual_triplet.choice:
        return 1
    else:
        return 0
