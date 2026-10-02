import numpy as np

from openmm.app import *
from openmm import *
from openmmtools.integrators import VVVRIntegrator
from openmm import unit
from openmm.unit import *

import mdtraj as md
from mdtraj.reporters import DCDReporter

import h5py

from .AlwaysAcceptingTPS import TPS_sampler
from . import mcg
import scipy.constants as constants
import os
import re
import fcntl
import time
import shutil
import sys
import glob

def cv(x_p, L):

    co2_coms, water_coms, box_length = mcg.CO2ClathrateCOM(x_p, L)
    MCG_OP, _, _ = mcg.MCG_optimized(co2_coms, water_coms, box_length)

    return MCG_OP

def stateFunction(x, box_length):
    c = cv(x, box_length)
    # print(s,c)

    if c <= 10:
        return "A", c 
    elif c >= 300:
        return "B", c
    else:
        return "0", c


## this is needed to safely name data folders squentially while starting multiple scripts at the same time
def get_next_run_folder(base_dir="data", lock_file="folder_lock.lock"):

    print(f"[PID {os.getpid()}] Calling get_next_run_folder")

    lock_path = os.path.join(base_dir, lock_file)

    # Open a lock file to coordinate access
    with open(lock_path, "w") as lock:
        # Wait for and acquire exclusive lock
        fcntl.flock(lock, fcntl.LOCK_EX)

        # Double-check once locked: get next available number
        existing = [d for d in os.listdir(base_dir) if re.match(r"run_\d+$", d)]
        numbers = [int(re.search(r"run_(\d+)", d).group(1)) for d in existing]
        next_number = max(numbers) + 1 if numbers else 0
        new_folder = os.path.join(base_dir, f"run_{next_number}")
        os.makedirs(new_folder)
        print(f"[PID {os.getpid()}] Created folder: {new_folder}")


        # Release lock automatically when closing
        return next_number, new_folder




def run_aaa_tps():

    print(f"[PID {os.getpid()}] Starting run_aaa_tps()")


    mu_gnd = 175
    alpha_gnd = 15
    beta_gnd = 2.5


    base_path = args.base_path
    os.makedirs(base_path, exist_ok = True)

    next_number, data_folder = get_next_run_folder(base_path)
    
    
    print(f"Created new folder: {data_folder}")

    total_trials = 50


    T = 260
    P = 500

    initial_traj_folder = args.initial_traj_folder
    initial_traj_idx = 20

    gro_folder = "gro_files"

    # Get the trajectory input from the matching folder
    traj_folder = os.path.join(initial_traj_folder, f"run_{next_number}")
    traj_files = glob.glob(os.path.join(traj_folder, "traj_*.dcd"))
    
    print(f'PATH {traj_folder}')

    if len(traj_files) != 1:
        raise FileNotFoundError(f"Expected exactly one traj_*.dcd file in {traj_folder}, found {len(traj_files)}")

    input_file = traj_files[0]
    print(f"Using trajectory file: {input_file}")

    #shutil.copy(input_file, os.path.join(data_folder, os.path.basename(input_file)))

    input_coordinates=os.path.join(gro_folder, 'conf.gro')
    input_topology=os.path.join(gro_folder, 'topol.top')

    positions_mdtraj = md.load(input_file, top=input_coordinates)


    print(f'Loading input coordinates from: {input_coordinates}\nLoading input topology from: {input_topology}')

    #### LOADING OF FILES AND SIMULATION #####################################################

    try:
        gro = GromacsGroFile(input_coordinates)
        top = GromacsTopFile(input_topology, periodicBoxVectors=gro.getPeriodicBoxVectors())
    except:
        print("Problem loading files...")
        print("ending script...")

    
    # Define the target pressure and temperature
    pressure = P * bar
    temperature = T * kelvin

    # Define the integration timestep and collision rate
    timestep = 2.0 * femtoseconds
    collision_rate = 1.0 / picoseconds

    # Calculate the frequency in terms of integration steps
    barostat_frequency = int((4 * picoseconds) / timestep)  # This will be 2000 steps


    system = top.createSystem(nonbondedMethod = PME, nonbondedCutoff = 1 * nanometer, constraints = HBonds)  # Using particle Mesh Ewald, setting nonbonded radius to  1nm and constraining only Hydrogen bonds
    system.addForce(MonteCarloBarostat(pressure, temperature, barostat_frequency)) # Pressure, Temperature, frequency
    system.addForce(CMMotionRemover())
    integrator = VVVRIntegrator(temperature, collision_rate, timestep) 
    simulation_eq = Simulation(top.topology, system, integrator)


   # initial_traj = md.load_hdf5(input_file, stride = 1)
    initial_traj = md.load(input_file, top=input_coordinates)


    # Allocate velocity array
    velocities = np.zeros((initial_traj.n_frames, initial_traj.n_atoms, 3))
    
    for i in range(initial_traj.n_frames):
        # Set positions for this frame
        simulation_eq.context.setPositions(initial_traj.xyz[i])
        
        # Resample velocities at the desired temperature
        simulation_eq.context.setVelocitiesToTemperature(temperature)
    
        # Get the state and extract velocities
        state = simulation_eq.context.getState(getVelocities=True)
        vel = state.getVelocities(asNumpy=True)  # Quantity array (nm/ps)
    
        # Convert to ndarray in units of m/s or whatever you want
        velocities[i] = vel.value_in_unit(nanometers / picoseconds)
    
    # Assign to traj (note: mdtraj doesn't track units)
    initial_traj.velocities = velocities

    
    simulation_eq.context.setPositions(initial_traj.xyz[initial_traj_idx])

    #with h5py.File(input_file, "r") as f:
    #    velocities = f["velocities"][:]

    #initial_traj.velocities = velocities


    #simulation_eq.context.setVelocities(velocities[initial_traj_idx] * nanometers/picosecond)  # Ensure units
    simulation_eq.context.setVelocitiesToTemperature(temperature)
    simulation_eq.context.setPeriodicBoxVectors(*initial_traj.unitcell_vectors[initial_traj_idx])

    state = simulation_eq.context.getState()
    current_vectors = state.getPeriodicBoxVectors()

    simulation_eq.reporters.append(StateDataReporter(stdout, 1e5, step=True, potentialEnergy=True, temperature=True, volume = True))


    sampler = TPS_sampler(simulation_eq, data_folder, mu_gnd, alpha_gnd, beta_gnd)

    sampler.sample(initial_traj, stateFunction, cv, total_trials, stride=1e5, maxPathLength=3e8) #mpl in unit steps

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run AAA-TPS for one CO2 clathrate hydrate replica.",
    )
    p.add_argument(
        "--initial-traj-folder", required=True,
        help="Parent folder containing run_N/traj_*.dcd initial paths.",
    )
    p.add_argument(
        "--base-path", required=True,
        help="Output folder. A new run_N subfolder is created inside it.",
    )
    p.add_argument(
        "--temperature", type=float, default=260.0,
        help="Temperature in K (default: 260).",
    )
    p.add_argument(
        "--pressure", type=float, default=500.0,
        help="Pressure in bar (default: 500).",
    )
    p.add_argument(
        "--c-ref", type=float, default=175.0,
        help="Reference MGC-1 for the shooting-point weight (default: 175).",
    )
    p.add_argument(
        "--gn-alpha", type=float, default=15.0,
        help="Generalized normal scale alpha (default: 15).",
    )
    p.add_argument(
        "--gn-beta", type=float, default=2.5,
        help="Generalized normal shape beta (default: 2.5).",
    )
    p.add_argument(
        "--n-trials", type=int, default=1000,
        help="Number of TPS trials (default: 1000).",
    )
    p.add_argument(
        "--initial-traj-idx", type=int, default=20,
        help="Frame index in the initial path to shoot from first "
             "(default: 20).",
    )
    p.add_argument(
        "--gro-folder", default="gro_files",
        help="Folder containing conf.gro and topol.top (default: gro_files).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_aaa_tps(
        initial_traj_folder=args.initial_traj_folder,
        base_path=args.base_path,
        temperature=args.temperature,
        pressure=args.pressure,
        c_ref=args.c_ref,
        gn_alpha=args.gn_alpha,
        gn_beta=args.gn_beta,
        n_trials=args.n_trials,
        initial_traj_idx=args.initial_traj_idx,
        gro_folder=args.gro_folder,
    )


if __name__ == "__main__":
    main()