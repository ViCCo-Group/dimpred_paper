# %% Packages and functions and global variables.
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from helper import (
    correlate_embeddings,
    create_plot_order_for,
    evaluate_prediction_from,
)

from utils.utils import (
    determine_base_path,
    load_pickle,
    noise_ceiling_group,
    rgb2hex,
    save_pickle,
)

base_path = determine_base_path()
data_path = f"{base_path}/dimpred_paper/data"
results_path = f"{base_path}/dimpred_paper/results"
mpl.rcParams["svg.fonttype"] = "none"
palette2 = ["#1f78b4", "#ff7f00", "#33a02c"]
own_palette = [
    rgb2hex(104, 176, 205),
    rgb2hex(71, 72, 149),
    rgb2hex(161, 49, 65),
    rgb2hex(226, 94, 50),
    rgb2hex(57, 114, 206),
    rgb2hex(175, 220, 215),
]
save_plot = False
imgformat = "pdf"

# %% Import data.
indices_200ref = np.loadtxt(f"{data_path}/raw/humans/indices_200ref.txt").astype(int)

# Note: "kriegeskorte-118" is actually "cichy-118". This image set was misnomered
# initially and the misnomer escalated throughout all scripts and data objects.
# The label "kriegeskorte-118" has only been changed posthoc in the manuscript but
# not in any script.
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

noise_ceiling = {}
for imgset in ["kriegeskorte-92", "kriegeskorte-118"]:
    single_rms = np.load(
        f"{data_path}/raw/ground_truth_representational_matrices/dissimilarity_{imgset}_single.npy"
    )
    noise_ceiling[imgset] = noise_ceiling_group(single_rms)

crsa_results = load_pickle(f"{data_path}/processed/crsa_all_processed.pkl")
frrsa_rsm_corrs = load_pickle(f"{data_path}/processed/frrsa_all_processed.pkl")
dimpred_results = load_pickle(f"{data_path}/processed/dimpred_all_processed.pkl")

# %% I. Analyze how well each DNN-module predicts individual dimensions by
# correlating the predicted embedding with the true embedding. only done
# for imgset=='1854ref' because only here exists the true_embedding.
# Recreates Figure S1 in the Supplementary Information.
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

fig = plt.figure()
fig.set_size_inches(25, 15)
ax = sns.lineplot(data=embedding_corrs_truth, legend=True)
ax.set(
    xlabel="SPoSE Dimension",
    ylabel="Pearson's r between predicted and actual Dimension Vector",
    title="Different DNN's performance across SPoSE dimensions",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
if save_plot:
    save_name = "dnn_predicted_vs_true_dims.pdf"
    plt.savefig(f"{results_path}/{save_name}", dpi=200, bbox_inches="tight")
    print(f"Printed {save_name}")
else:
    plt.show()
plt.close()

print(f"Top 5 DNNs on average: \n {embedding_corrs_truth.mean().head(5)}")
print(f"Top 5 DNNs on best dimension: \n {embedding_corrs_truth.max().head(5)}")

# %% II. Correlate dimpred-RSM and cRSA-RSMs with ground-truth RSM.
# Combine dimpred, crsa and frrsa scores in one pd frame and plot.
dimpred_rsm_corrs = evaluate_prediction_from("dimpred", dimpred_results, true_rsms)
crsa_rsm_corrs = evaluate_prediction_from("crsa", crsa_results, true_rsms)
all_rsm_corrs_wide = dimpred_rsm_corrs.merge(frrsa_rsm_corrs, how="outer").merge(
    crsa_rsm_corrs, how="outer"
)
save_pickle(all_rsm_corrs_wide, f"{results_path}/all_rsm_corrs_wide")

all_rsm_corrs_long = pd.melt(
    all_rsm_corrs_wide,
    id_vars=["imgset", "model_module"],
    value_vars=["dimpred_r", "frrsa_r", "crsa_r"],
    var_name="method",
    value_name="correlation",
)
order_method = ["dimpred_r", "frrsa_r", "crsa_r"]
all_rsm_corrs_long["order_method"] = all_rsm_corrs_long.apply(
    lambda row: create_plot_order_for(row, "method", order_method), axis=1
)

# %% III. Overview pointplot all DNNs for imgset=1854ref.
# Recreates Figure 3 of the manuscript ("Ensemble Top-8" in Figure 3 was place manually)
cond = all_rsm_corrs_long["imgset"] == "1854ref"
df = all_rsm_corrs_long[cond].sort_values(by=["order_method", "correlation"])

fig = plt.figure()
fig.set_size_inches(15, 7)
ax = sns.pointplot(
    data=df,
    x="model_module",
    y="correlation",
    hue="method",
    dodge=False,
    join=False,
    markers="d",
    palette=[own_palette[1], own_palette[0], own_palette[2]],
)
ax.set_xticklabels(
    ax.get_xticklabels(), rotation=45, horizontalalignment="right", fontsize="x-small"
)
ax.set(
    title=f"All DNNs for imageset {imgset}",
    xlabel="DNN Architecture",
    ylabel="Pearson's r between predicted and ground-truth RSM",
)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
handle1 = mpatches.Patch(color=own_palette[1], label="DimPred")
handle2 = mpatches.Patch(color=own_palette[0], label="FR-RSA")
handle3 = mpatches.Patch(color=own_palette[2], label="cRSA")
plt.legend(
    handles=[handle1, handle2, handle3],
    title="Method",
    bbox_to_anchor=(0.1, 1.0),
    fancybox=True,
    edgecolor="white",
    borderaxespad=0.0,
)
if save_plot:
    save_name = f"overview_{imgset}.{imgformat}"
    plt.savefig(f"{results_path}/{save_name}", dpi=200, bbox_inches="tight")
    print(f"Printed {save_name}")
else:
    plt.show()
plt.close()

# %% IV. Print stats for every imageset.
# Recreates some statistics reported in various subsections of the manuscript's result's section, namely:
# "Accurate prediction of perceived similarity for broadly-sampled natural object images", first paragraph.
# "DimPred generalizes to out-of-set images and other similarity tasks", second paragraph.
# "Predictive accuracy depends on granularity of an image set’s similarity structure", second paragraph.
for imgset in all_rsm_corrs_wide.imgset.unique():
    cond = all_rsm_corrs_wide["imgset"] == imgset
    df = all_rsm_corrs_wide[cond]
    dimpred_best = df.apply(
        lambda x: True
        if ((x["dimpred_r"] > x["frrsa_r"]) and (x["dimpred_r"] > x["crsa_r"]))
        else False,
        axis=1,
    )
    x = len(dimpred_best[dimpred_best == True].index)
    y = len(dimpred_best.index)
    print("==========================")
    print("==========================")
    print(f"IMAGESET: {imgset}:")
    print(f"Dimpred is better than frrsa _and_ crsa in {x} of {y} cases.")

    n = 5
    best_n_dnn = df.nlargest(n, "dimpred_r").loc[:, ["model_module", "dimpred_r"]]
    print(f"Best {n} DNNs based on absolute Dimpred-score are:")
    print(best_n_dnn)

# %% V. Scatterplot validation sets.
# Recreates graphs in Figures 4 and 6 of the manuscript ("Ensemble model" stars were place manually).
mini = -0.1
for kind in ["heterogeneous", "homogeneous"]:
    for comparison in ["frrsa", "crsa"]:
        if kind == "heterogeneous":
            cond = all_rsm_corrs_wide["imgset"].isin(
                [
                    "48new",
                    "48nonref",
                    "kriegeskorte-118",
                    "kriegeskorte-92",
                    "peterson-various",
                ]
            )
        elif kind == "homogeneous":
            cond = all_rsm_corrs_wide["imgset"].isin(
                [
                    "peterson-animals",
                    "peterson-automobiles",
                    "peterson-fruits",
                    "peterson-furniture",
                    "peterson-vegetables",
                ]
            )
        df = all_rsm_corrs_wide[cond]
        alpha = 0.95
        s = 40
        fig = plt.figure()
        fig.set_size_inches(5, 5)
        ax = sns.scatterplot(
            x=df[f"{comparison}_r"],
            y=df["dimpred_r"],
            alpha=alpha,
            s=s,
            hue=df["imgset"],
            style=df["imgset"],
            palette=own_palette[:5],
            edgecolor="w",
        )
        ax.set(
            ylim=(mini, 0.9),
            xlim=(mini, 0.9),
            xlabel=f"{comparison}",
            ylabel="DimPred",
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.plot(
            [mini, 0.9], [mini, 0.9], linewidth=2, color="black", linestyle="dashed"
        )
        if save_plot:
            save_name = f"overview_scatter_{kind}_{comparison}.{imgformat}"
            plt.savefig(f"{results_path}/{save_name}", dpi=200, bbox_inches="tight")
            print(f"Printed {save_name}")
        else:
            plt.show()
        plt.close()

    # Print some statistics reported in various subsections of the manuscript's result's section, namely:
    # Caption of Figure 4.
    # "Predictive accuracy depends on granularity of an image set’s similarity structure", second paragraph (and caption of Figure 6)
    dimpred_better_crsa = len((df[df["dimpred_r"] > df["crsa_r"]]))
    print(
        f"For {kind} imagesets, dimpred is better than crsa in {dimpred_better_crsa} of {len(df)} cases."
    )
    dimpred_better_frrssa = len((df[df["dimpred_r"] > df["frrsa_r"]]))
    print(
        f"For {kind} imagesets, dimpred is better than frrsa in {dimpred_better_frrssa} of {len(df)} cases."
    )
