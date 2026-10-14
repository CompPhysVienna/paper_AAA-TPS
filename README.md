# An Always Accepting Algorithm for Transition Path Sampling (AAA-TPS)
Companion repository for the paper "An Always Accepting Algorithm for Transition Path Sampling" aka "AAA-TPS"

## Installation
The necessary software for the execution of the notebooks and the instruction for its installation can be found in the folder `conda_envs`. In particular see the header of the file `spec-file_aaa_tps.txt`.

To install the package with the TPS algorithms, open a terminal in the main folder and run `pip install -e .` which installs the package in edit mode. In this way there is no need to reinstall the package every time the source is modified.  

## Contents
- `aaa_tps/`: Python package with the TPS algorithms (two-way shooting, one-way shooting, ARA-TPS and AAA-TPS) used for the two-dimensional models.
- `2D_models/`: simulations, analysis and figures for the two-dimensional models (Fig. 2 of the main text, Figs. S1, S2, S6 and Table S1 of the Supplementary Material). See `2D_models/README.md`.
- `Clathrates/`: AAA-TPS code and figure source data for CO₂ clathrate hydrates. See `Clathrates/README.md`.

Do not hesitate to open issues for bugs or questions.
