# dimpred_paper

This repository holds all analyses code to reproduce results of the _paper_ accompanying `DimPred`. The toolbox `dimpred` itself resides in a [separate repository](https://github.com/PhilippKaniuth/dimpred) (which is installed into this project as an editable source package (via `pip install -e .`)).

Most directories and subdirectories have their own `README.md` explaining their content further.

`data` and `results` can be found on [OSF](https://osf.io/jtekq/).

## The directory structure

```
├── README.md
├── data                <-  .gitignored due to size.
│   ├── images          <- All image sets used in this project.
│   ├── interim         <- Intermediate data based on raw data that has been transformed using scripts.
│   ├── processed       <- The final data forming the basis for results.
│   └── raw             <- Original and immutable data serving as input to scripts.
│
├── results             <- Output from scripts that go into the manuscript. .gitignored due to size.
│
├── scripts             <- Interactive scripts used in this project that perform a specific end-level job.
│
├── .gitignore
├── LICENSE.md
├── environment.yml
```

## Which figure was produced by which script?
- Figure 1: handmade using Inkscape.
- Figure 2: handmade using Inkscape.
- Figure 3: `compute_dnn_results.py`, cell III.
- Figure 4: `compute_dnn_results.py`, cell V.
- Figure 5: `compute_human_results`, cell IV, (see also `compute_dnn_ensemble_model_results.py`, cell II)
- Figure 6: `compute_dnn_results.py`, cell V.
- Figure 7: `visualize_heatmaps.py`
- Figure 8: Made by [O. Contier](https://github.com/oliver-contier).

- Figure S1: `compute_dnn_results.py`, cell I.
- Table S2: `compute_dnn_ensemble_model_results.py`, cell II.
- Figure S3: `compute_human_results.py`, cell I.

For redundancy, each script has comments in each relevant cell explicating which figure (and which stats where in the paper) it produces.

## How to reproduce
1. Clone this repository to your local machine.
2. Download `data` and `results` from [OSF](https://osf.io/jtekq/) and put them inside this directory as indicated by [the directory structure](#the-directory-structure).
3. Run ...