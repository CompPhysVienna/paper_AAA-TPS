#!/bin/bash
# Template for the cluster runs of the convergence test on the bi-stable double well (Fig. S2).
# Each task runs one batch of replicas (simulation_<task id>) with run_bsdw_tps_simulation.py.
# Adapt account, partition, time and environment to your cluster and submit from the 2D_models folder:
#
#   sbatch scripts/submit_simulations.slurm AAA-TPS Gaussian
#   sbatch scripts/submit_simulations.slurm twoway_shooting Uniform --symmetrize-histo    # reference
#
# In the paper, 32 batches of 75 replicas (2400 replicas) were run for each method and
# 329 batches of 75 replicas (24 675 replicas, several submissions) for the reference.

#SBATCH --job-name=tps_sims
#SBATCH --output=logs/sim_%j.out
#SBATCH --error=logs/sim_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
##SBATCH --account=<account>
##SBATCH --partition=<partition>

ALGORITHM=$1
P_SEL=$2
shift 2

# Activate the environment with the aaa_tps package (see conda_envs/)
# conda activate aaa_tps

srun --ntasks=$SLURM_NTASKS --cpus-per-task=1 \
    bash -c "python -u scripts/run_bsdw_tps_simulation.py \$SLURM_PROCID --algorithm $ALGORITHM --p-sel $P_SEL $*"
