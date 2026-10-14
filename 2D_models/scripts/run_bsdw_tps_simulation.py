#!/usr/bin/env python3
"""
Run one batch of independent TPS replicas on the bi-stable double well, saving
only the p(x|TP) histogram of each replica (no trajectories).

This is the production script used for the convergence test of the bi-stable
double well (Fig. 2 of the Supplementary Material). Each call runs
`--n-replicas` replicas sequentially in

    <output-root>/<algorithm>/bsdw/<p_sel>/simulation_<simulation_id>/R_<replica>/

and is meant to be launched many times in parallel (one per core), see
`submit_simulations.slurm`.

The reference ensemble is generated with two-way shooting, uniform selection
probability and `--symmetrize-histo`, which adds the reflection x^1 -> -x^1 of
every sampled configuration to the histogram.

Usage (from the 2D_models folder):
    python scripts/run_bsdw_tps_simulation.py <simulation_id> --algorithm AAA-TPS --p-sel Gaussian
"""

import os
import time
import argparse
import numpy as np
from functools import partial

from aaa_tps.systems.two_dimensional_double_wells import BistableDoubleWell
from aaa_tps.states.states import circular_state
from aaa_tps.utils.flexible_length_tps import generate_path
from aaa_tps.utils.selection_probabilities import UniformSelection, GaussianSelection
from aaa_tps.cvs.cvs import linear_cv
from aaa_tps.TPS_algorithms.twoway_shooting import TwoWayShootingTPS
from aaa_tps.TPS_algorithms.oneway_shooting import OneWayShootingTPS
from aaa_tps.TPS_algorithms.ARA_TPS import ARA_TPS
from aaa_tps.TPS_algorithms.AAA_TPS import AAA_TPS

ALGORITHMS = {
    "twoway_shooting": TwoWayShootingTPS,
    "oneway_shooting": OneWayShootingTPS,
    "ARA-TPS": ARA_TPS,
    "AAA-TPS": AAA_TPS,
}


def main():
    parser = argparse.ArgumentParser(description="Run a batch of TPS replicas on the bi-stable double well.")
    parser.add_argument("simulation_id", type=int, help="Index of this batch of replicas (e.g. $SLURM_PROCID).")
    parser.add_argument("--algorithm", choices=ALGORITHMS.keys(), required=True)
    parser.add_argument("--p-sel", choices=["Uniform", "Gaussian"], required=True,
                        help="Shooting point selection probability.")
    parser.add_argument("--symmetrize-histo", action="store_true",
                        help="Symmetrize the p(x|TP) histogram along x^1 (used for the reference).")
    parser.add_argument("--n-replicas", type=int, default=75, help="Replicas run sequentially by this batch.")
    parser.add_argument("--total-trials", type=int, default=102500, help="TPS trials per replica (including equilibration).")
    parser.add_argument("--equilibration-trials", type=int, default=2500, help="Trials discarded at the beginning of each replica.")
    parser.add_argument("--init-path", default="./output/twoway_shooting/bsdw/Gaussian/final_path_2w.txt",
                        help="Initial reactive path (default: final path of the two-way shooting run "
                             "of bistable_double_well_generate_data.ipynb).")
    parser.add_argument("--output-root", default="./output/cluster", help="Root output folder.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (default: from the current time). The seed of the batch is seed + simulation_id.")
    args = parser.parse_args()

    print(f"Simulation {args.simulation_id:03d} on host {os.uname().nodename}", flush=True)

    # Random seed: different for every batch
    base_seed = args.seed if args.seed is not None else int(time.time() * 1e6) % (2**32)
    np.random.seed((base_seed + args.simulation_id) % (2**32))

    # System and stable states (see Supplementary Material)
    bsdw = BistableDoubleWell(A=15, B=0.25)
    bsdw_state_A = circular_state(center=(-2., 0), axes=(1, 1), angle=0, radius=0.15)
    bsdw_state_B = circular_state(center=(2., 0), axes=(1, 1), angle=0, radius=0.15)

    # Overdamped Langevin dynamics
    beta = 1
    timestep = 0.001
    diffusion_coefficient = 1
    max_path_length = 25000

    generate = partial(generate_path, force_function=bsdw.force_function,
                                      beta=beta,
                                      timestep=timestep,
                                      diffusion_coefficient=diffusion_coefficient)

    if args.p_sel == "Uniform":
        p_sel = UniformSelection()
    else:
        p_sel = GaussianSelection(cv=linear_cv(a=1, b=0), k=12.5, cv_ref=0)

    current_path = np.loadtxt(args.init_path)

    output_dir = os.path.join(args.output_root, args.algorithm, "bsdw", args.p_sel, f"simulation_{args.simulation_id:03d}")

    tps_simulation = ALGORITHMS[args.algorithm](max_path_length=max_path_length,
                                                state_A=bsdw_state_A,
                                                state_B=bsdw_state_B,
                                                generate_path=generate,
                                                p_sel=p_sel,
                                                init_path=current_path,
                                                output_dir=output_dir,
                                                n_replicas=args.n_replicas,
                                                symmetrize_histo=args.symmetrize_histo)

    for replica in range(args.n_replicas):
        tps_simulation.run_tps_simulation(args.total_trials,
                                          configuration_output_frequency=1,
                                          path_length_output_frequency=1,
                                          path_output_frequency=1,
                                          force_evaluation_output_frequency=1,
                                          dump_output_frequency=0,
                                          equilibration_trials=args.equilibration_trials,
                                          show_progress=False,
                                          dump_paths=False,
                                          dump_histo=True,
                                          replica=replica)

        print(f"Simulation {args.simulation_id:03d}: completed replica {replica + 1} out of {args.n_replicas}", flush=True)


if __name__ == "__main__":
    main()
