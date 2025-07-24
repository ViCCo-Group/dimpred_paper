#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes one comprehensive pd.df for each (cRSA, FR-RSA, DimPred, and DimRating).

This module is only used when new analysis results for one or more methods have
been created. It contains a wrapper for and the method-specific functions doing
the hard work and can be used from the CLI.

For cRSA, this module calls the function that direcly conducts cRSA using raw
DNN data and transforms them to processed cRSA data.
For FR-RSA and DimPred this module takes interim data and transforms them to
processed data.
For DimRating, raw human experimental data is loaded and transformed to
processed data.

@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from utils.utils import (
    determine_base_path,
    euclidian_rdm,
    load_pickle,
    pearson_rdm,
    pearson_rsm,
    save_pickle,
    spose_rsm,
)

base_path = determine_base_path()
data_path = f"{base_path}/dimpred_paper/data"


def get_parameters_for(file_name):
    """Based on the file name, determine various parameters."""
    indx = [m.start() for m in re.finditer("_", file_name)]
    imageset = file_name[indx[-1] + 1 : -4]
    module = file_name[indx[-2] + 1 : indx[-1]]
    model = file_name[indx[-3] + 1 : indx[-2]]
    try:
        regularization = file_name[indx[-4] + 1 : indx[-3]]
        ndim = file_name[indx[-5] + 1 : indx[-4]]
    except IndexError:
        regularization, ndim = None, None
    return model, module, imageset, regularization, ndim


def load_crsa_from(path):
    """Create and load cRSA RMs for each DNN module.

    For each DNN module, different classical RSA representational matrices are
    computed. Together with additional information, everything is put in a
    pd.df.

    Parameters
    ----------
    path : str
        Directory which contains one folder for each DNN which in turn contains
        one subfolder for each module (`thingsvision` output structure.)

    Returns
    -------
    dnn_rsms : pd.DataFrame
        Holds the different RMs and various additional information. Columns
        are as follows:

        =============   =========================================================
        model           The DNN architecture (as `str`)
        module          The specific module of the DNN (as `str`)
        model_module    `model` and `module` fused (as `str`)
        imageset        The imageset for which embedding was predicted (as `str`)
        regularization  None
        ndim            None
        embedding       The predicted SPoSE embedding (as `nd.array`)
        rsm_spose       None
        rsm_pearson     Pearson RSM based on `embedding` (as `nd.arary`)
        rdm_pearson     Pearson RDM based on `embedding` (as `nd.array`)
        rdm_euclidean   Euclidean RDM based on `embedding` (as `nd.array`)
        =============   =========================================================
    """
    models = os.listdir(path)
    dnn_rsms = pd.DataFrame(columns=[])
    for model in models:
        if model == "vgg19-bn-matconvnet":  # exclude matlab version of VGG19.
            continue
        print(model)
        modules = os.listdir(f"{path}/{model}")
        for module in modules:
            print(module)
            imagesets = os.listdir(f"{path}/{model}/{module}")
            for imageset in imagesets:
                if imageset == "THINGS":
                    continue
                print(imageset)
                try:
                    embedding = np.loadtxt(
                        f"{path}/{model}/{module}/{imageset}/features.txt"
                    )
                except (FileNotFoundError, IOError, OSError):
                    embedding = np.loadtxt(
                        f"{path}/{model}/{module}/{imageset}/features-srp.txt"
                    )
                rsm_pearson = pearson_rsm(embedding)
                rdm_pearson = pearson_rdm(embedding)
                rdm_euclidean = euclidian_rdm(embedding)
                dnn_rsms_i = pd.DataFrame(
                    {
                        "model": model,
                        "module": module,
                        "model_module": f"{model}_{module}",
                        "imageset": imageset,
                        "regularization": None,
                        "ndim": None,
                        "embedding": [embedding],
                        "rsm_spose": None,
                        "rsm_pearson": [rsm_pearson],
                        "rdm_pearson": [rdm_pearson],
                        "rdm_euclidean": [rdm_euclidean],
                    }
                )
                dnn_rsms = pd.concat([dnn_rsms, dnn_rsms_i], ignore_index=True)
    return dnn_rsms


def load_frrsa_from(path):
    """Load FR-RSA scores for each DNN module.

    For each DNN module, the scores resulting from applying feature-
    reweighted RSA are loaded. Together with additional information,
    everything is put in a pd.df.

    Parameters
    ----------
    path : str
        Directory where the individual FR-RSA score files live.

    Returns
    -------
    scores : pd.DataFrame
        Holds the scores and various additional information. Columns
        are as follows:

        ============   =======================================================
        imageset       The imageset for which frrsa was performed (as `str`)
        model_module   The DNN module for which frrsa was performed (as `str`)
        frrsa_r        The correlation score as returned by frrsa (as `float`)
        ============   =======================================================
    """
    files = glob.glob(path)
    scores = pd.DataFrame(columns=[])
    for file_name in files:
        model, module, imageset, *_ = get_parameters_for(file_name)
        if "matconvnet" in model:
            continue
        scores_i = load_pickle(file_name)
        scores_i.loc[0, "imageset"] = imageset
        scores_i.loc[0, "model_module"] = f"{model}_{module}"
        scores_i.rename(columns={"score": "frrsa_r"}, inplace=True)
        scores_i.drop(["target"], axis=1, inplace=True)
        scores = pd.concat([scores, scores_i], ignore_index=True)
    return scores


def load_dimpred_from(path):
    """Load dimpred for each DNN module.

    For each DNN module, the predicted embedding is loaded. Then,
    different RSMs/RDMS based on that embedding are computed. Everything
    is put into a pd.df.

    Parameters
    ----------
    path : str
        Directory where the individual predicted embeddings.

    Returns
    -------
    dimpred_dnn_results : pd.DataFrame
        Holds the predicted embedding, RSMs, and various information
        pertaining to that specific anayses. Columns are as follows:

        =============   =========================================================
        model           The DNN architecture (as `str`)
        module          The specific module of the DNN (as `str`)
        model_module    `model` and `module` fused (as `str`)
        imageset        The imageset for which embedding was predicted (as `str`)
        regularization  Regularization scheme used when predicting (as `str`)
        ndim            Dimensionality of predicted SPoSE embedding (as `int`)
        embedding       The predicted SPoSE embedding (as `nd.array`)
        rsm_spose       SPoSE RSM based on `embedding` (as `nd.array`)
        rsm_pearson     Pearson RSM based on `embedding` (as `nd.arary`)
        rdm_pearson     Pearson RDM based on `embedding` (as `nd.array`)
        rdm_euclidean   Euclidean RDM based on `embedding` (as `nd.array`)
        =============   =========================================================
    """
    files = glob.glob(path)
    dimpred_results = pd.DataFrame(columns=[])
    for i, file_name in enumerate(files):
        print(f"..currently file {i} from {len(files)}...")
        model, module, imageset, regularization, ndim = get_parameters_for(file_name)
        embedding = np.loadtxt(file_name)
        rsm_spose = spose_rsm(embedding)
        rsm_pearson = pearson_rsm(embedding)
        rdm_pearson = pearson_rdm(embedding)
        rdm_euclidean = euclidian_rdm(embedding)
        dimpred_results_i = pd.DataFrame(
            {
                "model": model,
                "module": module,
                "model_module": f"{model}_{module}",
                "imageset": imageset,
                "regularization": regularization,
                "ndim": ndim,
                "embedding": [embedding],
                "rsm_spose": [rsm_spose],
                "rsm_pearson": [rsm_pearson],
                "rdm_pearson": [rdm_pearson],
                "rdm_euclidean": [rdm_euclidean],
            }
        )
        dimpred_results = pd.concat(
            [dimpred_results, dimpred_results_i], ignore_index=True
        )
    return dimpred_results


def load_dimrating_from(path, idx_200ref):
    """Load rating task data and transform into embedding.

    Load the data from the first experiment (i.e., the rating task) and
    calculate one embedding across participants for all presented image sets.

    Parameters
    ----------
    path : str
        Path to the file which holds the filtered rating task data.
    idx_200ref : np.ndarray
        Holds the indices of the 200-ref image set as integers.

    Returns
    -------
    predicted_embeddings : dict
        A dictionary holding for each image set the human generated embedding
        as an np.ndarray.
    predicted_rsms_type :
        A dictionary holding for each image set the SPoSE RSM based on the
        human generated embedding as an np.ndarray.
    """
    human_data = pd.read_csv(f"{path}", delimiter=",", header=0)
    n_dim = 49
    image_codes = {}
    image_codes["48nonref"] = range(2000, 2048)
    image_codes["48new"] = range(3000, 3048)
    image_codes["200ref"] = idx_200ref
    predicted_embeddings = {}
    predicted_rsms = {}
    for imageset in image_codes.keys():
        predicted_embeddings[imageset] = np.zeros((len(image_codes[imageset]), n_dim))
        for dim in human_data.dim_id.unique():
            for idx, image in enumerate(image_codes[imageset]):
                # "dim-1" in the following necessary because human_data.dim_id is a Matlab index.
                predicted_embeddings[imageset][idx, dim - 1] = np.mean(
                    human_data.loc[
                        (human_data["img_code"] == image)
                        & (human_data["dim_id"] == dim),
                        "dim_score_rated",
                    ]
                )
        predicted_rsms[imageset] = spose_rsm(predicted_embeddings[imageset])
    return (predicted_embeddings, predicted_rsms)


def process_data_for(method):
    """Wrapper function that calls a method-specific data loader.

    Each method (crsa, frrsa, dimpred) has its own custom data loader
    function. These funcs go through all respective files and create
    one big pd.df. This pd.df is saved to disk as a pickle file.
    Supposed to be called from the command line.

    Parameters
    ----------
    method : str
        Denotes the method of which the results shall be loaded and saved.
    """
    print("Start collecting...")
    allowed = ["crsa", "frrsa", "dimpred", "dimrating"]
    assert method in allowed, "\nIllegal argument\n"

    file_name = f"{data_path}/processed/{method}_all_processed"
    assert not bool(os.path.isfile(f"{file_name}.pkl")), f"{file_name} exists already. \
        Double-check and delete prior to repeating this command."

    if method == "crsa":
        processed = load_crsa_from(f"{data_path}/raw/dnns")
    elif method == "frrsa":
        processed = load_frrsa_from(f"{data_path}/interim/frrsa/*")
    elif method == "dimpred":
        processed = load_dimpred_from(f"{data_path}/interim/dimpred/predictions*")
    elif method == "dimrating":
        idx_200ref = np.loadtxt(f"{data_path}/raw/humans/indices_200ref.txt").astype(
            int
        )
        path = f"{data_path}/raw/humans/exp1_rating_task_filtered.csv"
        processed = load_dimrating_from(path=path, idx_200ref=idx_200ref)

    save_pickle(processed, file_name)

    print("Collecting completed!")


if __name__ == "__main__":
    process_data_for(sys.argv[1])
