#!/usr/bin/env python3
"""
Average the p(x|TP) histograms of all the replicas of a cluster run.

Histograms are searched in base_dir/RUN_*/simulation_*/R_*/P_x_TP*.txt
(the layout produced by collect_simulations.py). The average is normalized to
one and saved to base_dir/avg_P_x_TP.txt, together with a log file.

This is used to build the reference p(x|TP) for the convergence test of the
bi-stable double well.

Usage:
    python average_histograms.py <base_dir>
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm


def main():
    if len(sys.argv) != 2:
        print("Usage: python average_histograms.py <base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory.")
        sys.exit(1)

    run_dirs = sorted(d for d in os.listdir(base_dir) if d.startswith("RUN_"))

    sum_hist = None
    found_histograms = 0
    total_histograms = 0
    failed_paths = []

    for run_dir in tqdm(run_dirs, desc="Processing RUN folders", unit="run"):
        run_path = os.path.join(base_dir, run_dir)
        sim_dirs = sorted(d for d in os.listdir(run_path)
                          if d.startswith("simulation_") and os.path.isdir(os.path.join(run_path, d)))

        for sim_dir in sim_dirs:
            sim_path = os.path.join(run_path, sim_dir)
            replica_dirs = sorted(r for r in os.listdir(sim_path)
                                  if r.startswith("R_") and os.path.isdir(os.path.join(sim_path, r)))

            for replica in replica_dirs:
                total_histograms += 1
                matches = sorted(glob.glob(os.path.join(sim_path, replica, "P_x_TP*.txt")))
                try:
                    hist = np.loadtxt(matches[0])
                except Exception:
                    failed_paths.append(os.path.join(sim_path, replica))
                    continue

                if sum_hist is None:
                    sum_hist = np.zeros_like(hist, dtype=np.float64)
                sum_hist += hist
                found_histograms += 1

    if found_histograms == 0:
        print("No histograms found. Check your folder structure or file names.")
        sys.exit(1)

    avg_hist = sum_hist / found_histograms
    avg_hist /= np.sum(avg_hist)

    avg_path = os.path.join(base_dir, "avg_P_x_TP.txt")
    np.savetxt(avg_path, avg_hist)

    log_path = os.path.join(base_dir, "avg_P_x_TP.log")
    with open(log_path, "w") as log:
        log.write(f"Base directory: {base_dir}\n")
        log.write(f"Runs processed: {len(run_dirs)}\n")
        log.write(f"Total histograms expected: {total_histograms}\n")
        log.write(f"Total histograms found: {found_histograms}\n")
        log.write(f"Average histogram saved to: {avg_path}\n\n")
        if failed_paths:
            log.write("Missing or unreadable histograms:\n")
            for path in failed_paths:
                log.write(f"  {path}\n")

    print(f"Averaged {found_histograms} histograms across {len(run_dirs)} runs.")
    print(f"Output: {avg_path}")
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
