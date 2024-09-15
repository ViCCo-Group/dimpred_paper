#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize heatmaps created by Florian Mahner.

F.M. created heatmaps for specific image sets which reside in "results-interim-
heatmaps". These heatmaps can be visualized with this script. Functions in this
script were written by F.M. with minor changes by P.K.

@author: Florian Mahner (mahner@cbs.mpg.de)
@author: Philipp Kaniuth (kaniuth@cbs.mpg.de)
"""

# %%
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from utils.utils import determine_base_path

base_path = determine_base_path()
plot_path = f"{base_path}/dimpred_paper/results"


def compute_aggregate_saliency(
    saliency_maps: np.ndarray, dimension_values: np.ndarray
) -> np.ndarray:
    aggregate_saliency = np.zeros_like(saliency_maps[0])
    for smap, dim_val in zip(saliency_maps, dimension_values):
        aggregate_saliency += smap * dim_val
    aggregate_saliency /= np.sum(dimension_values)
    return aggregate_saliency


def plot_heatmaps(
    paths: list, iteration: int, save_plot: bool, plot_path: str, imgformat: str
):
    max_rows = 5
    n_maps = len(paths)
    fig = plt.figure(
        figsize=(8, 32)
    )  # The ratio between these two numbers must correspond to the ratio of max_rows and n_maps. If anything (e.g. text labels) is overlapping, increase these two numbers but keep the ratio.
    gs1 = gridspec.GridSpec(n_maps, max_rows)
    gs1.update(wspace=0.025, hspace=0.3)  # set the spacing between axes.

    for i, path in enumerate(paths):
        results = pickle.load(open(path, "rb"))

        img = results["img"]
        img_name = results["query"]
        argsort_dims = results["argsort_dims"]
        argsort_vals = results["argsort_dimvals"]
        saliency = results["saliency"]
        ax = plt.subplot(gs1[i, 0])
        ax.imshow(img)
        ax.set_title(f"{img_name}", fontsize=10)

        ax = plt.subplot(gs1[i, 1])
        # sort saliency by topk_dims

        cmap = cv2.COLORMAP_JET
        img = (img * 255).astype(np.uint8)

        sorted_saliency = saliency[argsort_dims]
        aggregate_saliency = compute_aggregate_saliency(sorted_saliency, argsort_vals)

        aggregate_saliency = (aggregate_saliency - aggregate_saliency.min()) / (
            aggregate_saliency.max() - aggregate_saliency.min()
        )
        aggregate_saliency = (aggregate_saliency * 255).astype(np.uint8)
        aggregate_saliency = cv2.bitwise_not(aggregate_saliency)
        heatmap_img = cv2.applyColorMap(aggregate_saliency, cmap)
        super_imposed_img = cv2.addWeighted(heatmap_img, 0.4, img, 0.6, 0)

        ax.imshow(super_imposed_img)
        ax.set_title("Relevance for similarity prediction", fontsize=8)

        for j in range(2, max_rows):
            ax = plt.subplot(gs1[i, j])
            saliency_dim = saliency[argsort_dims[j - 2]]

            saliency_dim = (saliency_dim - saliency_dim.min()) / (
                saliency_dim.max() - saliency_dim.min()
            )
            saliency_dim = (saliency_dim * 255).astype(np.uint8)
            saliency_dim = cv2.bitwise_not(saliency_dim)
            heatmap_img = cv2.applyColorMap(saliency_dim, cmap)
            super_imposed_img = cv2.addWeighted(heatmap_img, 0.4, img, 0.6, 0)

            ax.imshow(super_imposed_img)
            value = argsort_vals[j - 2]
            dim = argsort_dims[j - 2]
            ax.set_title(f"Dim {dim+1}: {value:.2f}", fontsize=8)

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Iteration: {iteration}", fontsize=16)

    if save_plot:
        save_name = f"heatmaps.{imgformat}"
        plt.savefig(f"{plot_path}/{save_name}", dpi=300, transparent=True)
        print(f"Printed {save_name}")
    else:
        plt.show()
    plt.close()


# %% Plot heatmaps.
# Recreates graphs from Figure 7.
selected_images = [
    "chalkboard",
    "fire_pit",
    "quad",
    "seatbelt",
]

images_paths = [
    f"{base_path}/dimpred_paper/data/processed/heatmaps/THINGSplus/6000/8/0.1/{imgage}.pkl"
    for imgage in selected_images
]

stop = int(np.round(len(images_paths) / 20))
if stop == 0:
    stop = 1

for i in range(0, stop, 1):
    start = i * 20
    end = (i + 1) * 20
    if end >= len(images_paths):
        end = len(images_paths)
    current_imgs = images_paths[start:end]
    plot_heatmaps(
        paths=current_imgs,
        iteration=i,
        save_plot=False,
        plot_path=plot_path,
        imgformat="pdf",
    )
