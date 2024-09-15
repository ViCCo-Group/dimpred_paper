# %% Packages and functions and global variables.
import numpy as np
import pandas as pd
from helper import (
    correlate_embeddings,
    select_dnns_per_dim,
)

from utils.utils import (
    correlate_matrices,
    determine_base_path,
    euclidian_rdm,
    load_pickle,
    save_pickle,
    spose_rsm,
)

base_path = determine_base_path()
data_path = f"{base_path}/dimpred_paper/data"
results_path = f"{base_path}/dimpred_paper/results"

# %% Import data.
indices_200ref = np.loadtxt(f"{data_path}/raw/humans/indices_200ref.txt").astype(int)

true_rsms = {}
imagesets = [
    "48nonref",
    "48new",
    "1854ref",
    "200ref",
    "peterson-various",
    "peterson-animals",
    "peterson-automobiles",
    "peterson-fruits",
    "peterson-furniture",
    "peterson-vegetables",
    "kriegeskorte-92",
    "kriegeskorte-118",
]
for imgset in imagesets:
    if "peterson" in imgset:
        true_rsms[imgset] = np.loadtxt(
            f"{data_path}/raw/ground_truth_representational_matrices/similarity_{imgset}.txt"
        )
    elif "kriegeskorte" in imgset:
        true_rsms[imgset] = np.loadtxt(
            f"{data_path}/raw/ground_truth_representational_matrices/dissimilarity_{imgset}.txt"
        )
    elif imgset == "200ref":
        true_rsms[imgset] = np.loadtxt(
            f"{data_path}/raw/ground_truth_representational_matrices/similarity_49d_1854ref.txt"
        )[indices_200ref, :][:, indices_200ref]
    else:
        true_rsms[imgset] = np.loadtxt(
            f"{data_path}/raw/ground_truth_representational_matrices/similarity_49d_{imgset}.txt"
        )
true_embedding = np.loadtxt(f"{data_path}/raw/original_spose/embedding_49d.txt")

dimpred_results = load_pickle(f"{data_path}/processed/dimpred_all_processed.pkl")

# %% I. Analyze how well each DNN-module predicts individual dimensions by
# correlating the predicted embedding with the true embedding. only done
# for imgset=='1854ref' because only here exists the true_embedding.
embedding_corrs_truth = {}
cond1 = dimpred_results["imgset"] == "1854ref"
model_modules = dimpred_results[cond1].model_module.unique()
for model_module in model_modules:
    cond2 = cond1 & (dimpred_results["model_module"] == model_module)
    predicted_embedding = dimpred_results.loc[cond2, "embedding"].tolist()[0]
    embedding_corrs_truth[model_module] = correlate_embeddings(
        predicted_embedding, true_embedding
    )
embedding_corrs_truth = pd.DataFrame.from_dict(embedding_corrs_truth)
embedding_corrs_truth = embedding_corrs_truth.reindex(
    sorted(embedding_corrs_truth.columns), axis=1
)

# %% II. Select the best n DNNs per dimension and then combine dimensions.
# Recreates Table S2 in the Supplementary Information.
# Computes "ensemble_model_rsms" which is used in "compute_human_results.py" that eventually recreates Figure 5.
# Computes "ensemble_model_scores" which is used in "compute_human_results.py" that uses it to compute statistics.
imagesets = [
    "48nonref",
    "48new",
    "1854ref",
    "peterson-various",
    "peterson-animals",
    "peterson-automobiles",
    "peterson-fruits",
    "peterson-furniture",
    "peterson-vegetables",
    "kriegeskorte-92",
    "kriegeskorte-118",
]

imgs_per_set = [48, 48, 1854, 120, 120, 120, 120, 120, 120, 92, 118]

n_dnn = len(embedding_corrs_truth.axes[1])
best_dnns_per_dim = select_dnns_per_dim(embedding_corrs_truth, n_dnn)
combined_corrs = {}
list_length = n_dnn * len(imagesets)
combined_corrs["r"] = np.zeros(list_length)
combined_corrs["imgset"] = [None] * list_length
combined_corrs["i_dnn"] = np.zeros(list_length)
combined_dnn_rsms = {}
for imgset in imagesets:
    combined_dnn_rsms[imgset] = {}

for i_dnn in range(n_dnn):
    id_length = i_dnn * len(imagesets)
    combined_embeddings = {}
    for i_imgset, imgset in enumerate(imagesets):
        n_imgs = imgs_per_set[i_imgset]
        combined_embeddings[imgset] = np.zeros((n_imgs, 49))
        for dim in range(49):
            dnn_combo = best_dnns_per_dim.loc[dim].tolist()[: i_dnn + 1]
            combined_vector = np.zeros((n_imgs))
            for model_module in dnn_combo:
                cond = (dimpred_results["imgset"] == imgset) & (
                    dimpred_results["model_module"] == model_module
                )
                combined_vector += dimpred_results.loc[cond, "embedding"].tolist()[0][:, dim]  # fmt: skip
            combined_vector /= len(dnn_combo)
            combined_embeddings[imgset][:, dim] = combined_vector
        if "kriegeskorte" in imgset:
            combined_dnn_rsms[imgset][i_dnn + 1] = euclidian_rdm(
                combined_embeddings[imgset]
            )
        else:
            combined_dnn_rsms[imgset][i_dnn + 1] = spose_rsm(
                combined_embeddings[imgset]
            )
        r_n, _ = correlate_matrices(
            true_rsms[imgset], combined_dnn_rsms[imgset][i_dnn + 1]
        )
        combined_corrs["r"][id_length + i_imgset] = r_n
        combined_corrs["imgset"][id_length + i_imgset] = imgset
        combined_corrs["i_dnn"][id_length + i_imgset] = i_dnn + 1
combined_corrs = pd.DataFrame.from_dict(combined_corrs)

save_pickle(combined_dnn_rsms, f"{results_path}/ensemble_model_rsms")
combined_corrs.sort_values(by=["imgset", "i_dnn"], inplace=True)
combined_corrs = combined_corrs.pivot(index="imgset", columns="i_dnn", values="r")
combined_corrs.to_csv(f"{results_path}/ensemble_model_scores.csv", sep=",")
