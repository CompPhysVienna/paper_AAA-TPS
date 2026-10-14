# Two-dimensional models

Simulations and analysis of the two-dimensional models: the standard double well (`stdw`) and the bi-stable
double well (`bsdw`). This folder reproduces Fig. 2 of the main text and Figs. S1, S2 and S6 and Table S1 of the
Supplementary Material.

## Quick start: figures from the processed data

The processed data used in the paper are included in `reference/` and `results/`. To reproduce the figures run

- `figures.ipynb` → `figures/Fig2_bistable_double_well.png`, `figures/FigS1_standard_double_well.png`,
  `figures/FigS2_bistable_double_well_convergence.png`, `figures/FigS6_path_length_distributions.png`

The numbers of Table S1 are in `results/efficiency_stdw.txt` and `results/efficiency_bsdw.txt`.

## Full workflow: from the simulations

The raw simulation output (trajectories, several hundred GB in total) is written to `output/`, which is not
tracked. The workflow is:

| Step | Notebook / script | Output | Figure |
|---|---|---|---|
| 1 | `standard_double_well_generate_data.ipynb` | `output/<algorithm>/stdw/<p_sel>/R_xxxx/` | |
| 2 | `standard_double_well_analysis.ipynb` | `reference/twoway_shooting/stdw/Uniform/`, `results/<algorithm>/stdw/`, `results/path_lengths/stdw_P_L.txt`, `results/efficiency_stdw.txt` | S1, S6, Table S1 |
| 3 | `bistable_double_well_generate_data.ipynb` | `output/<algorithm>/bsdw/<p_sel>/R_xxxx/` | |
| 4 | `bistable_double_well_analysis.ipynb` | `reference/twoway_shooting/bsdw/Uniform/`, `results/<algorithm>/bsdw/` (`ch_correlation`), `results/path_lengths/bsdw_P_L.txt`, `results/efficiency_bsdw.txt` | 2B, 2D, S6, Table S1 |
| 5 | `bistable_double_well_path_overlap.ipynb` | `results/<algorithm>/bsdw/` (`OPL_vs_feval`, `OPL_renewal`) | 2C |
| 6 | `scripts/` (cluster) | `results/<algorithm>/bsdw/` (`convergence_l1`, `convergence_l1_sym`), `reference/twoway_shooting/bsdw/Uniform/avg_P_x_TP.txt` | S2 |
| 7 | `figures.ipynb` | `figures/` | 2, S1, S2, S6 |

The generation notebooks run, for each model, 50 replicas of $10^5$ trials (after $2.5\cdot10^3$ equilibration
trials) for each method and 250 replicas of two-way shooting with uniform selection probability for the
reference ensemble. The trajectories are needed by the analysis notebooks (`dump_paths=True`).

Folder names: `twoway_shooting` (two-way shooting, run name `2w`), `oneway_shooting` (one-way shooting, `1w`),
`ARA-TPS` / `ARA_TPS` (always-reactive algorithm), `AAA-TPS` / `AAA_TPS` (always-accepting algorithm);
`Gaussian` and `Uniform` refer to the shooting point selection probability. The reference ensemble is
`twoway_shooting/<model>/Uniform`.

### Convergence test for the bi-stable double well (Fig. S2)

The convergence test uses 2400 replicas per method and 24 675 replicas for the reference, which only store the
$p(x|\mathrm{TP})$ histogram of each replica. These runs were performed on a cluster with the scripts in `scripts/`
(run from the `2D_models` folder, after running `bistable_double_well_generate_data.ipynb` to obtain the initial
path `output/twoway_shooting/bsdw/Gaussian/final_path_2w.txt`):

```bash
mkdir -p logs
# Reference: two-way shooting, uniform selection, histograms symmetrized along x^1
sbatch scripts/submit_simulations.slurm twoway_shooting Uniform --symmetrize-histo
python scripts/collect_simulations.py output/cluster/twoway_shooting/bsdw/Uniform   # after each submission
python scripts/average_histograms.py output/cluster/twoway_shooting/bsdw/Uniform

# Methods (32 batches of 75 replicas each)
sbatch scripts/submit_simulations.slurm AAA-TPS Gaussian    # and similarly for the other methods

# Convergence of the running average (plain and symmetrized along x^1)
REF=output/cluster/twoway_shooting/bsdw/Uniform/avg_P_x_TP.txt
python scripts/convergence_histograms.py output/cluster/AAA-TPS/bsdw/Gaussian $REF
python scripts/convergence_histograms.py output/cluster/AAA-TPS/bsdw/Gaussian $REF --symmetrize
```

`convergence_histograms.py` writes `convergence_l1.txt` and `convergence_l1_sym.txt` in the method folder; copy
them to `results/<algorithm>/bsdw/<p_sel>/convergence_l1[_sym]_<run>.txt` for `figures.ipynb`.

## Notes

- AAA-TPS is reweighted with the trajectory weights `acc_w` for uniform selection and `1/acc_p_sel_norm` for
  Gaussian selection.
- The convergence curves of AAA-TPS in Fig. S1C use a random resampling of the trajectories according to their
  weights, so re-running the analysis gives statistically equivalent but not identical curves.
- The trajectories needed by the analysis notebooks are large (up to ~2.4 GB per replica for the bi-stable double
  well); the notebooks load one replica at a time.
- Computing the path overlap chains (`bistable_double_well_path_overlap.ipynb`) takes several hours per method.
