# dimpred_paper

This repository holds all analyses code to reproduce results of the manuscript. The toolbox `DimPred` itself resides in a [separate repository](https://github.com/ViCCo-Group/dimpred).

Most directories and subdirectories have their own `README.md` explaining their content further.

`data` and `results` can be found on [OSF](https://osf.io/jtekq/).

## The directory structure

```
├── README.md
├── data                <- .gitignored. Download from OSF.
│   └── processed       <- The final data forming the basis for results.
│
├── results             <- Output from scripts that go into the manuscript. .gitignored. Download from OSF.
│
├── scripts             <- Interactive scripts used in this project that perform a specific end-level job.
│
├── .gitignore
├── LICENSE.md
├── environment.yml
```

## Which figure in the manuscript was produced by which script?
- Figure 1: handmade using Inkscape.
- Figure 2: handmade using Inkscape.
- Figure 3: `compute_dnn_results.py`, cell III.
- Figure 4: `compute_dnn_results.py`, cell V.
- Figure 5: `compute_human_results`, cell IV, (see also `compute_dnn_ensemble_model_results.py`, cell II).
- Figure 6: `compute_dnn_results.py`, cell V.
- Figure 7: `visualize_heatmaps.py`
- Figure 8: Made by [O. Contier](https://github.com/oliver-contier).

- Figure S1: `compute_dnn_results.py`, cell I.
- Table S2: `compute_dnn_ensemble_model_results.py`, cell II.
- Figure S3: `compute_human_results.py`, cell I.

For redundancy, each script has comments in each relevant cell explicating which figure (and which stats where in the paper) it produces.

## How to reproduce
1. Clone this repository to your local machine (e.g., via `git clone https://github.com/ViCCo-Group/dimpred_paper`)
2. Download `data` and `results` from [OSF](https://osf.io/jtekq/), unpack them and put them inside your local version of this directory as indicated by [the directory structure](#the-directory-structure). You need to download _all_ files from OSF to be able to reproduce the findings.
3. Create an environment using this repository's [environment.yml](https://github.com/ViCCo-Group/dimpred_paper/blob/main/environment.yml). For example, if you have `conda` on your system, execute `conda env create --file=/PATH/TO/environment.yml`. Replace `/PATH/TO/environment.yml` with the actual path to the environment file.
4. Activate the environment and execute those scripts you are interested in.

## :page_with_curl: Citation
If you use any of this code or the manuscript, please cite our [associated manuscript](https://www.biorxiv.org/content/10.1101/2024.06.28.601184v2) as follows:

```
@preprint{Kaniuth_Dimpred,
	author = {Kaniuth, Philipp and Mahner, Florian P. and Perkuhn, Jonas and Hebart, Martin N.},
	title = {A high-throughput approach for the efficient prediction of perceived similarity of natural objects},
	year = {2024},
	doi = {10.1101/2024.06.28.601184},
	journal = {bioRxiv},
	url = {https://www.biorxiv.org/content/10.1101/2024.06.28.601184v2},
	journal = {bioRxiv}
}
```
