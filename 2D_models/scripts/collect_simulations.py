#!/usr/bin/env python3
"""
Move the completed batches of a cluster run into a new RUN_xx folder.

A batch (simulation_nnn folder produced by run_bsdw_tps_simulation.py) is
considered completed if its first replica contains the p(x|TP) histogram.
Completed batches are moved to base_dir/RUN_xx/ and renumbered from 000, so that
several submissions writing to the same base_dir can be accumulated.

Usage:
    python collect_simulations.py <base_dir>
"""

import os
import sys
import glob
import shutil


def main():
    if len(sys.argv) != 2:
        print("Usage: python collect_simulations.py <base_dir>")
        sys.exit(1)

    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a valid directory.")
        sys.exit(1)

    # First free RUN_xx folder
    run_index = 0
    while os.path.exists(os.path.join(base_dir, f"RUN_{run_index:02d}")):
        run_index += 1
    run_dir = os.path.join(base_dir, f"RUN_{run_index:02d}")
    os.makedirs(run_dir)

    log_path = os.path.join(base_dir, f"collect_successful_run_{run_index:02d}.log")

    sim_dirs = sorted(d for d in os.listdir(base_dir)
                      if d.startswith("simulation_") and os.path.isdir(os.path.join(base_dir, d)))

    successful = []
    failed = []

    for sim_dir in sim_dirs:
        sim_path = os.path.join(base_dir, sim_dir)
        replicas = sorted(r for r in os.listdir(sim_path)
                          if r.startswith("R_") and os.path.isdir(os.path.join(sim_path, r)))

        if replicas and glob.glob(os.path.join(sim_path, replicas[0], "P_x_TP*.txt")):
            successful.append(sim_dir)
        else:
            failed.append(sim_dir)

    # Move completed batches and renumber them
    for new_idx, sim_dir in enumerate(successful):
        shutil.move(os.path.join(base_dir, sim_dir), os.path.join(run_dir, f"simulation_{new_idx:03d}"))

    with open(log_path, "w") as log:
        log.write(f"Base dir: {base_dir}\n")
        log.write(f"Output run folder: {run_dir}\n")
        log.write(f"Total simulations found: {len(sim_dirs)}\n")
        log.write(f"Successful: {len(successful)}\n")
        log.write(f"Failed: {len(failed)}\n\n")
        log.write("Successful simulations moved:\n")
        for sim in successful:
            log.write(f"  {sim}\n")
        log.write("\nFailed simulations:\n")
        for sim in failed:
            log.write(f"  {sim}\n")

    print(f"{len(successful)} successful simulations moved to {run_dir}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
