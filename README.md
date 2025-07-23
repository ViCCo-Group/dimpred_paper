# :star2: DimPred - Similarities via Embeddings

This is the code repository of the project `A high-throughput approach for the efficient prediction of perceived similarity of natural objects`.


It holds all analysis code needed to reproduce results presented in the [paper](https://elifesciences.org/reviewed-preprints/105394v1). For all data needed to reproduce the manuscript's results please see the project's data repository on [OSF](https://osf.io/jtekq/).


The toolbox `DimPred` itself resides in a [separate repository](https://github.com/ViCCo-Group/dimpred).


## :open_file_folder: The directory structure

```
├── README.md
├── data                <- .gitignored. Download from OSF.
│   ├── images          <- All image sets used in this project.
│   ├── interim         <- Intermediate data based on raw data that has been transformed using scripts.
│   ├── processed       <- The final data forming the basis for results.
│   └── raw             <- Original and immutable data serving as input to scripts.
│
├── results             <- Output from scripts that go into the manuscript. .gitignored. Download from OSF.
│
├── scripts             <- Interactive scripts used in this project that perform a specific end-level job.
│
├── .gitignore
├── LICENSE
└── environment.yml
```

## :bar_chart: Which figure in the manuscript was produced by which script?
| Content  | Script |
| ------------- | ------------- |
| Figure 1  |  handmade using Inkscape  |
| Figure 2  | handmade using Inkscape |
| Figure 3 | `compute_dnn_results.py`, cell III |
| Figure 4 | `compute_dnn_results.py`, cell V |
| Figure 5 | `compute_human_results`, cell IV, (see also `compute_dnn_ensemble_model_results.py`, cell II) |
| Figure 6 | `compute_dnn_results.py`, cell V |
| Figure 7 | `visualize_heatmaps.py` |
| Figure 8 | Made by [O. Contier](https://github.com/oliver-contier) |
| Figure S1 | `compute_dnn_results.py`, cell I|
| Table S2  | `compute_dnn_ensemble_model_results.py`, cell II|
| Figure S3 | `compute_human_results.py`, cell I|

For redundancy, each script has comments in each relevant cell explicating which figure (and which stats where in the paper) it produces.

## :mag: How to reproduce the paper's finding
### Preparation
0. Read this section completely first.
1. Clone this repository to your local machine (e.g., via `git clone https://github.com/ViCCo-Group/dimpred_paper`).
2. Download all the files from [OSF](https://osf.io/jtekq/), unpack them and put them inside your local version of this directory as indicated by [the directory structure](#the-directory-structure).
3. Read on.

### Caveats
- Since we are not allowed to share the ground-truth representational matrices for {Cichy-118, Peterson-Automobiles, Peterson-Fruits, Peterson-Furniture,  Peterson-Vegetables, Peterson-Various}, the respective files in `data-raw-ground_truth_representational_matrices` contain dummy data. This is to not brake the scripts when you execute them, but this of course makes all the results you reproduce for these image sets different from the actual results. We are also not allowed to share the respective image sets.
- For each computational model used in this project, the directory `data-raw-dnns` holds activations extracted for each image set using the Python package `thingsvision`. Note that the respective directory you received when having downloaded everything from the data repository on OSF is actually empty: the files together would be very large and can deterministically be recreated by running `thingsvision` with all the parameters mentioned in the manuscript's supplementary information if you want to.
- You can choose to reproduce the manuscript's findings on different levels, where the first one is the most elaborate and the last one the quickest.

#### Level I
On this level, you will re-extract all activations from all computational models used in this study and run `dimpred` and `frrsa` to reproduce all basic output files. Specifically:

0. Install [`thingsvision`](https://github.com/ViCCo-Group/thingsvision). Using all the parameters mentioned in the manuscript's supplementary information, extract activations for each computational model and public image set and save those activations into `data-raw-dnns`. You should create a separate conda environment for `thingsvision`.

1. Run [`dimpred`](https://github.com/PhilippKaniuth/dimpred) and [`frrsa`](https://github.com/ViCCo-Group/frrsa/tree/master) for all computational models and put the output into `data-interim-dimpred/frrsa`. You should create separate conda environments for each.

#### Level II.
On this level, you assume `data-interim-dimpred` and `data-interim-frrsa` to be fine, but you will reproduce the {}_all_processed.pkl files in `data-processed`.

2. Run the script `transform_to_processed.py` which outputs the {}_all_processed.pkl files. Note that these already exist in `data-processed` because they are needed on Level III (but will be overwritten if you execute this step). Without executing step 0, you can execute this step for everything except `method == "crsa"` since that method acts directly on the raw model activations (see the docstrings in `transform_to_processed.py`).

#### Level III.
This level is likely the one you want to enter on. If you start here, you do not need to execute anything from prior steps. On this level, you will need to create an environment using this repository's [environment.yml](https://github.com/ViCCo-Group/dimpred_paper/blob/main/environment.yml).

3. Run `compute_dnn_results.py` to reproduce reported statistics and figures (and to reproduce all_rsm_corrs_wide.pkl which already lives in `data-processed` and is used in step 5. but will be overwritten if you execute this step).

4. Run `compute_dnn_ensemble_model_results.py` to reproduce reported statistics and figures (and to reproduce ensemble_model_rsms.pkl which lives in `data-processed` and is used in step 5. but will be overwritten if you execute this step).

5. run `compute_human_results.py` to reproduce reported statistics and figures.

6. run `visualize_heatmaps.py` to reproduce reported figures.

Be aware of the [caveat note](https://github.com/ViCCo-Group/dimpred_paper/blob/main/scripts/CAVEAT.md).

<!-- Contact -->
## :wave: How to contact
In case of any questions or suggestions please reach out to either Philipp Kaniuth (kaniuth {at} cbs.mpg.de) or Martin Hebart (hebart {at} cbs.mpg.de).


<!-- License -->
## :warning: License
This GitHub repository is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE Version 3 - see the [LICENSE.md](LICENSE.md) file for details.


## :page_with_curl: Citation
If you use any of this code or the manuscript, please cite our [associated paper](https://elifesciences.org/reviewed-preprints/105394v1) as follows:

```
@article{Kaniuth_2025,
	author={Kaniuth, Philipp and Mahner, Florian P and Perkuhn, Jonas and Hebart, Martin N},
	title={A high-throughput approach for the efficient prediction of perceived similarity of natural objects},
	year={2025}, month=apr}
	DOI={10.7554/elife.105394.1},
	url={http://dx.doi.org/10.7554/eLife.105394.1},
	publisher={eLife Sciences Publications, Ltd}
}

```
