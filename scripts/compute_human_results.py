# %% Packages and functions and global variables.
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
from helper import (
    correlate_embeddings,
)

from utils.utils import (
    correlate_matrices,
    determine_base_path,
    load_pickle,
    rgb2hex,
)

base_path = determine_base_path()
data_path = f"{base_path}/dimpred_paper/data"
results_path = f"{base_path}/dimpred_paper/results"
mpl.rcParams["svg.fonttype"] = "none"
palette1 = ["#1b9e77", "#d95f02", "#7570b3"]
icefire = sns.color_palette("icefire", as_cmap=True)
own_palette = [rgb2hex(68, 43, 120), rgb2hex(106, 177, 205), rgb2hex(226, 95, 47)]
save_plot = False
imgformat = "pdf"

# %% Import data.
indices_200ref = np.loadtxt(f"{data_path}/raw/humans/indices_200ref.txt").astype(int)

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

true_rsms = {}
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
dimrating_embeddings, dimrating_rsms = load_pickle(
    f"{data_path}/processed/dimrating_all_processed.pkl"
)

ensemble_model_rsms = load_pickle(f"{results_path}/ensemble_model_rsms.pkl")
ensemble_model_scores = pd.read_csv(f"{results_path}/ensemble_model_scores.csv")
all_rsm_scores_wide = load_pickle(f"{results_path}/all_rsm_corrs_wide.pkl")

# %% I. For the 200 human ref images (subset of the 1854), for each dimension,
# plot association between human predicted and ground-truth values across all
# objects and calculate correlation.
# Recreates Figure S3 in the Supplementary Information (in which the respective DNN line was added manually.)
embedding_corrs_human = {}
embedding_corrs_human = correlate_embeddings(
    embedding1=dimrating_embeddings["200ref"],
    embedding2=true_embedding[indices_200ref, :],
)

fig = plt.figure()
fig.set_size_inches(25, 15)
ax = sns.lineplot(data=embedding_corrs_human, legend=True)
ax.set(
    xlabel="SPoSE Dimension",
    ylabel="Pearson's r between predicted and actual Dimension Vector",
    title="Human performance across SPoSE dimensions",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
if save_plot:
    save_name = "human_predicted_vs_true_dims.pdf"
    plt.savefig(f"{results_path}/{save_name}", dpi=200, bbox_inches="tight")
    print(f"Printed {save_name}")
else:
    plt.show()
plt.close()

print(f"Top performance for a dimension: \n {embedding_corrs_human.max()}")

# %% II.For each image set, correlate with ground-truth RSM either
# human-predicted RSMs
# combined human-predicted RSM with best individual DNN-predicted similarity.
# combined human-predicted RSM with DNN-ensemble similarity.

human_rsm_corrs = pd.DataFrame()
imagesets = ["48new", "48nonref"]
for i, imgset in enumerate(imagesets):
    true_rsm = true_rsms[imgset]
    human_rsm = dimrating_rsms[imgset]
    for target in ["human_only", "human_dnn", "human_ensemble"]:
        if target == "human_only":
            target_rsm = human_rsm
        elif target == "human_dnn":
            cond = (dimpred_results["imgset"] == imgset) & (
                dimpred_results["model_module"] == "OpenCLIP-RN50x64-openai_visual"
            )
            dnn_rsm = dimpred_results.loc[cond, "rsm_spose"].tolist()[0]
            target_rsm = (dnn_rsm + human_rsm) / 2
        elif target == "human_ensemble":
            dnn_rsm = ensemble_model_rsms[imgset][
                8
            ]  # 8 denotes the number of models combined into the ensemble model.
            target_rsm = (dnn_rsm + human_rsm) / 2
        r, p = correlate_matrices(true_rsm, target_rsm)
        human_rsm_corrs_i = pd.DataFrame(columns=[])
        human_rsm_corrs_i.loc[0, "imgset"] = imgset
        human_rsm_corrs_i.loc[0, "r"] = r
        human_rsm_corrs_i.loc[0, "p"] = p
        human_rsm_corrs_i.loc[0, "target"] = target
        human_rsm_corrs = pd.concat([human_rsm_corrs, human_rsm_corrs_i])
human_rsm_corrs.sort_values(by=["imgset", "r"], inplace=True)
human_rsm_corrs.reset_index(inplace=True, drop=True)


# %% III. Test the 4 relevant correlations differences.
# Since each correlation is based on the same amount of unique pairs of 48 objects n is always 1128.
# r is not normally distributed, so convert to Fisher's z score.
# Recreates statistics from the subsection "Humans can successfully rate images on embedding dimensions - but rely on information similar to those of DNNs" second paragraph.


def test_correlation_difference(r1, r2, n1, n2):
    z_r_1 = np.log((1 + r1) / (1 - r1)) / 2
    z_r_2 = np.log((1 + r2) / (1 - r2)) / 2

    SE_r_1 = 1 / np.sqrt(n1 - 3)
    SE_r_2 = 1 / np.sqrt(n2 - 3)

    z_r_diff = z_r_1 - z_r_2
    z = (
        z_r_diff / (SE_r_1 + SE_r_2)
    )  # assuming possibility of interdependence between samples and using Cauchy–Bunyakovsky–Schwarz inequality.
    p = sp.stats.norm.sf(np.abs(z)) * 2  # two-sided
    return p


imagesets = ["48new", "48nonref"]
for imgset in imagesets:
    print(f"Imageset: {imgset}")
    # Diff(corr(human_only), corr(human_dnn))
    cond1 = (human_rsm_corrs["imgset"] == imgset) & (
        human_rsm_corrs["target"] == "human_only"
    )
    r_human_only = human_rsm_corrs.loc[cond1, "r"].iloc[0]
    cond2 = (human_rsm_corrs["imgset"] == imgset) & (
        human_rsm_corrs["target"] == "human_dnn"
    )
    r_human_dnn = human_rsm_corrs.loc[cond2, "r"].iloc[0]
    p1 = test_correlation_difference(r_human_only, r_human_dnn, 1128, 1128)
    print(f"Correlation of Humans with ground-truth: {r_human_only}")
    print(f"Correlation of Humans-DNN-Combination with ground-truth: {r_human_dnn}")
    print(f"Human_only and Human&DNN are different with p_uncorrected = {p1}")

    # Diff(corr(best_dnn), corr(human_dnn))
    cond = (all_rsm_scores_wide["imgset"] == imgset) & (
        all_rsm_scores_wide["model_module"] == "OpenCLIP-RN50x64-openai_visual"
    )
    r_best_dnn = all_rsm_scores_wide.loc[cond, "dimpred_r"].iloc[0]
    print(f"Correlation of best DNN with ground-truth: {r_best_dnn}")
    p2 = test_correlation_difference(r_best_dnn, r_human_dnn, 1128, 1128)
    print(f"DNN_only and Human&DNN are different with p_uncorrected = {p2}")

    # Diff(corr(human_only), corr(human_ensemble))
    cond1 = (human_rsm_corrs["imgset"] == imgset) & (
        human_rsm_corrs["target"] == "human_ensemble"
    )
    r_human_ensemble = human_rsm_corrs.loc[cond1, "r"].iloc[0]
    print(f"Correlation of Human&Ensemble with ground-truth: {r_human_ensemble}")
    p3 = test_correlation_difference(r_human_only, r_human_ensemble, 1128, 1128)
    print(f"Human_only and Human&Ensemble are different with p_uncorrected = {p3}")

    # Diff(corr(ensemble), corr(human_ensemble))
    cond = ensemble_model_scores["imgset"] == imgset
    r_ensemble = ensemble_model_scores.loc[cond, "8.0"].iloc[0]
    print(f"Correlation of ensemble DNN with ground-truth: {r_ensemble}")
    p4 = test_correlation_difference(r_ensemble, r_human_ensemble, 1128, 1128)
    print(f"Ensemble and Human&Ensemble are different with p_uncorrected = {p4}")
    print("=========================")


# %% IV. Plot RSMs.
# Recreates plots in Figure 5 of the manuscript.
imagesets = ["48new", "48nonref"]
for imgset in imagesets:
    for target in ["ground_truth", "human_only", "human_dnn", "human_ensemble"]:
        if target == "ground_truth":
            target_rsm = true_rsms[imgset]
        elif target == "human_only":
            target_rsm = human_rsm
        elif target == "human_dnn":
            cond = (dimpred_results["imgset"] == imgset) & (
                dimpred_results["model_module"] == "OpenCLIP-RN50x64-openai_visual"
            )
            dnn_rsm = dimpred_results.loc[cond, "rsm_spose"].tolist()[0]
            target_rsm = (dnn_rsm + human_rsm) / 2
        elif target == "human_ensemble":
            target_rsm = ensemble_model_rsms[imgset][
                8
            ]  # 8 denotes the number of models combined into the ensemble model.
        rsv = sp.spatial.distance.squareform(target_rsm, force="tovector", checks=False)
        rsv_rank = sp.stats.rankdata(rsv)
        rsm_rank = sp.spatial.distance.squareform(
            rsv_rank, force="tomatrix", checks=False
        )
        np.fill_diagonal(rsm_rank, 991)

        plt.figure()
        plt.imshow(
            rsm_rank,
            cmap=icefire,
            norm=None,
            aspect="equal",
            interpolation=None,
            alpha=None,
            vmin=None,
            vmax=None,
            origin=None,
            extent=None,
            filternorm=1,
            filterrad=4.0,
            resample=None,
        )
        plt.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False,
            right=False,
            left=False,
            labelleft=False,
        )
        cb = plt.colorbar()
        cb.ax.yaxis.set_tick_params(labelright=False)
        plt.title(f"{target} RSM for {imgset}")
        if save_plot:
            save_name = f"{results_path}/human_rsm_{target}_{imgset}.{imgformat}"
            plt.savefig(save_name, dpi=200, bbox_inches="tight")
            print(f"Printed {save_name}")
        else:
            plt.show()
        plt.close()
