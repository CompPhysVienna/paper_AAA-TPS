#!/usr/bin/env python3
"""
Convergence of the p(x|TP) histograms of a cluster run towards a reference.

The histograms in base_dir/simulation_*/R_*/P_x_TP*.txt are averaged
incrementally (in the order of the simulation and replica folders). After each
replica, the normalized running average is compared with the normalized
reference histogram and the cumulative number of force evaluations is recorded
(from force_evaluations*.txt in the replica folder).

With --symmetrize, the running average is symmetrized along x^1
(p(x0, x1) -> [p(x0, x1) + p(x0, -x1)]/2) before the comparison, i.e. the
ensemble is artificially given exactly the same number of paths in the two
transition channels of the bi-stable double well.

Output: base_dir/convergence_<metric>[_sym].txt with columns
    count   metric_value   cumulative_force_evals

Usage:
    python convergence_histograms.py <base_dir> <ref_hist.txt> [--metric l1|js|emdf] [--symmetrize] [--no-progress]
"""

import os
import sys
import glob
import argparse
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance


def l1_distance(P, Q):
    """L1 (Manhattan) distance between two histograms."""
    return np.sum(np.abs(P - Q))


def js_divergence(P, Q, eps=1e-12):
    """Jensen-Shannon distance between two 2D histograms."""
    P = P.flatten() + eps
    Q = Q.flatten() + eps
    P /= np.sum(P)
    Q /= np.sum(Q)
    return jensenshannon(P, Q)


def emd_flattened(P, Q):
    """Earth Mover's Distance between the flattened histograms (fast approximation)."""
    P = P.flatten()
    Q = Q.flatten()
    P /= np.sum(P)
    Q /= np.sum(Q)
    return wasserstein_distance(P, Q)


METRICS = {"l1": l1_distance, "js": js_divergence, "emdf": emd_flattened}


def load_force_evaluations(replica_path):
    """Total number of force evaluations of a replica (0 if not available)."""
    matches = sorted(glob.glob(os.path.join(replica_path, "force_evaluations*.txt")))
    if not matches:
        return 0.0
    try:
        return np.sum(np.loadtxt(matches[0]))
    except Exception:
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="Compute convergence of 2D histograms vs reference.")
    parser.add_argument("base_dir", help="Base directory containing the simulation_* folders.")
    parser.add_argument("ref_hist", help="Reference histogram file.")
    parser.add_argument("--metric", choices=METRICS.keys(), default="l1", help="Convergence metric (default: l1).")
    parser.add_argument("--symmetrize", action="store_true", help="Symmetrize the running average along x^1.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar.")
    args = parser.parse_args()

    if not os.path.isdir(args.base_dir):
        print(f"Error: {args.base_dir} is not a valid directory.")
        sys.exit(1)
    if not os.path.isfile(args.ref_hist):
        print(f"Error: {args.ref_hist} not found.")
        sys.exit(1)

    metric_func = METRICS[args.metric]
    print(f"Using metric: {args.metric.upper()}{' (symmetrized)' if args.symmetrize else ''}")

    ref_hist = np.loadtxt(args.ref_hist)
    ref_hist = ref_hist / np.sum(ref_hist)

    sim_dirs = sorted(os.path.join(args.base_dir, d) for d in os.listdir(args.base_dir)
                      if d.startswith("simulation_") and os.path.isdir(os.path.join(args.base_dir, d)))
    replica_dirs = [os.path.join(sim_dir, r)
                    for sim_dir in sim_dirs
                    for r in sorted(os.listdir(sim_dir))
                    if r.startswith("R_") and os.path.isdir(os.path.join(sim_dir, r))]

    sum_hist = None
    count = 0
    total_force_evals = 0.0
    diffs = []
    missing_files = []

    for replica_path in tqdm(replica_dirs, desc="Processing histograms", unit="hist", disable=args.no_progress):
        matches = sorted(glob.glob(os.path.join(replica_path, "P_x_TP*.txt")))
        try:
            hist = np.loadtxt(matches[0])
        except Exception:
            missing_files.append(replica_path)
            continue

        total_force_evals += load_force_evaluations(replica_path)

        if sum_hist is None:
            sum_hist = np.zeros_like(hist, dtype=np.float64)
        sum_hist += hist
        count += 1

        avg_hist = sum_hist / count
        avg_hist /= np.sum(avg_hist)

        if args.symmetrize:
            avg_hist = (avg_hist + avg_hist[:, ::-1]) / 2

        diffs.append((count, metric_func(avg_hist, ref_hist), total_force_evals))

    suffix = f"{args.metric}{'_sym' if args.symmetrize else ''}"
    out_path = os.path.join(args.base_dir, f"convergence_{suffix}.txt")
    np.savetxt(out_path, diffs, fmt="%-10d %.8g %.1f", header="count   metric_value   cumulative_force_evals")

    log_path = os.path.join(args.base_dir, f"convergence_{suffix}_missing.log")
    with open(log_path, "w") as f:
        f.write(f"Total expected histograms: {len(replica_dirs)}\n")
        f.write(f"Total processed: {count}\n")
        f.write(f"Missing or unreadable histograms: {len(missing_files)}\n\n")
        for path in missing_files:
            f.write(f"{path}\n")

    print(f"Processed {count} histograms (out of {len(replica_dirs)})")
    print(f"Convergence data saved to: {out_path}")


if __name__ == "__main__":
    main()
