# Running the AAA-TPS workflow for CO₂ clathrate hydrates

# AAA-TPS for CO₂ clathrate hydrates

This subfolder accompanies the paper repository and contains the AAA-TPS
code and figure source data for the CO₂ clathrate hydrate application.
It is self-contained: it has its own `environment.yml`, its own
`src/` and `analysis_scripts/`, and its own `data/` with the CSVs behind
every figure. The environment described here is independent of the rest
of the parent repository.

For the theory and parameter choices, see the paper and its Supplementary
Material. This file only covers the practical steps.

---

## 0. Setup

### Environment

```bash
conda env create -f environment.yml
conda activate aaa-tps
```

### Build the modified GRADE

Cage counting uses a modified copy of GRADE (see
`GRADE_updated/MODIFICATIONS.md` for what changed and the upstream
attribution). The compiled binary is not shipped. Build it once:

```bash
cd GRADE_updated
make
cd ..
```

This produces `GRADE_updated/GRADE`. `analysis_scripts/GradeAnalysis.py`
expects it there.

### Directory overview

| Path | Purpose |
|---|---|
| `src/` | AAA-TPS sampler (`AlwaysAcceptingTPS.py`, `Clathrate_TPS.py`) |
| `gro_files/` | Topology and starting configuration for the clathrate system |
| `analysis_scripts/` | Post-processing: MGC-1, DCD→GRO conversion, cage counting |
| `GRADE_updated/` | Modified GRADE source (build with `make`) |
| `notebooks/` | Analysis and plotting notebooks |
| `data/` | Figure source data (one CSV per plotted series) |
| `figures/` | Output figures |

---

## 1. Run the TPS simulations

This section covers the 12-replica AAA-TPS production runs at
T = 260 K, P = 500 bar.

### 1.1 Prepare initial reactive paths

TPS needs at least one reactive trajectory connecting the liquid and solid
states. This repository does not ship initial paths; you must produce them
yourself (or obtain them from the authors). :)

An initial path must satisfy:

- the first frame is in the liquid state (MGC-1 < 10),
- the last frame is in the solid state (MGC-1 > 300),
- no intermediate frame lies in either stable state,
- consecutive frames are separated by an interval long enough that
  velocities have decorrelated. The paper uses 200 ps, which is safe
  for the CO₂ clathrate system.

## 1.2 Run AAA-TPS

From the repository root:

    python -m src.Clathrate_TPS \
        --initial-traj-folder /path/to/initial_paths \
        --base-path /path/to/output

`--initial-traj-folder` and `--base-path` are the only required flags.
Every other parameter defaults to the value used in the paper.

Optional flags can be passed on the command line to override the
defaults, for example:

    python -m src.Clathrate_TPS \
        --initial-traj-folder /path/to/initial_paths \
        --base-path /path/to/output \
        --n-trials 500 \
        --temperature 265

The full list of flags and their defaults is defined in the `parse_args`
function at the bottom of `src/Clathrate_TPS.py`. You can also print it
directly:

    python -m src.Clathrate_TPS --help

### Output layout

The script creates a new `run_N` folder inside `--base-path`. The
numbering matches the initial paths: if `--base-path` already contains
`run_0` and `run_1`, the next run is `run_2`, and the corresponding
initial path is read from

    <--initial-traj-folder>/run_2/traj_*.dcd

Concurrent launches against the same `--base-path` are safe. They
coordinate through a lock file, so each process gets its own `run_N`
and the matching initial path.

## 2. Count cages along the transition paths

This produces the cage-count data that feeds Fig. 3D and the
cage-distribution panels in the SI.

### 2.1 Convert `.dcd` to `.gro`

The sampler writes `.dcd` trajectories, but GRADE expects `.gro`. Convert one TPS run at a time:

    python analysis_scripts/DCD2GRO.py \
        --folder /path/to/output/run_0 \
        --savefolder /path/to/CageData/run_0

`--folder` points at a single TPS run directory (containing `run_*/` subfolders with `traj_*.dcd`), 
and `--savefolder` is where the `.gro` files land. `--gro-folder` defaults to `gro_files/` and only 
needs overriding if `conf.gro` lives elsewhere.

`--burst` (default 100000) must match the save stride used by the sampler in `Clathrate_TPS.py`. 
If you changed `stride` in `sampler.sample(...)`, pass the same value here.

Optional frame-subsampling knobs: `--stride 10` writes every 10th frame to the `.gro`; `-n 1` 
reads every frame from the `.dcd`. Both default to the paper's values.

### 2.2 Run GRADE

First build the modified GRADE binary (see §0).

Then, for each replica processed in §2.1:

    python analysis_scripts/GradeAnalysis.py \
        --savefolder /path/to/CageData/run_0

`--savefolder` must be the same directory that §2.1 wrote to — GRADE reads the `.gro` files from there and writes the per-frame cage counts alongside them.

`--grade-binary` defaults to `GRADE_updated/GRADE`; override it if you built the binary somewhere else.

`--stride` (default 10) must match the `--stride` you passed to `DCD2GRO.py` — it determines which `.gro` files are picked up.

Each frame produces per-cage counts, including the `5^12 6^2` (crystalline) and `4^1 5^10 6^2` (amorphous) cages used for channel classification.

