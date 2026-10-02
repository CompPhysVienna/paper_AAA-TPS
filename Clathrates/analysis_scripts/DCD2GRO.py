import argparse

def run(folder, savefolder, gro_folder, burst=100000, stride=10, n=1):
    """
    Convert TPS .dcd trajectories to .gro for GRADE.

    Parameters
    ----------
    folder : str
        TPS output folder containing run_*/traj_*.dcd.
    savefolder : str
        Where to write the .gro trajectories.
    gro_folder : str
        Folder containing conf.gro (used as topology).
    burst : int
        Number of MD integration steps between saved frames. Must match
        the ``stride`` used in Clathrate_TPS.py (default 1e5 = 200 ps).
    stride : int
        Subsample the trajectory every ``stride`` frames before saving.
    n : int
        Load stride, i.e. read every ``n``-th frame from the .dcd.
    """

    os.makedirs(savefolder, exist_ok = True)

    for run_folder in os.listdir(folder):
        run_folder_path = os.path.join(folder, run_folder)

        # Skip non-directories and hidden folders
        if not os.path.isdir(run_folder_path) or run_folder.startswith('.'):
            continue

        # Process trajectory files
        for succ_file in os.listdir(run_folder_path):
            if succ_file.startswith('traj_') and not succ_file.endswith('_0.dcd'):

                idx = succ_file.split('_')[1].split('.')[0]
                succ_file_path = os.path.join(run_folder_path, succ_file)
                # Input files

                mcg_file = os.path.join(run_folder_path, f'cv_{idx}.txt')

                mcg_frames, mcg_values = np.loadtxt(mcg_file, skiprows=1, unpack=True)


                cutoff_mask = mcg_values >= 300
                if np.any(cutoff_mask):
                    first_cutoff_idx = np.argmax(cutoff_mask)
                    B_cutoff = int( mcg_frames[first_cutoff_idx] / burst )
                else:
                    B_cutoff = None  # or -1 or np.nan if you prefer a sentinel value

                print(B_cutoff)
                topology_file = os.path.join(gro_folder, "conf.gro")


                # Load the trajectory with a stride of n (e.g., every 10th frame)
                n = 1  # Change this to the desired interval
                traj = md.load(succ_file_path, top=topology_file, stride=n)
                original_frame_indices = np.arange(traj.n_frames) * n

                traj.time = np.arange(0, traj.n_frames) * burst * n

                if len(mcg_frames) != len(traj):
                    print(error)
                    continue

                traj = traj[: B_cutoff + 1]

                stride = 10

                strided_traj = traj[::stride]

                # --- Rename 'HOH' residues to 'SOL' ---
                for residue in strided_traj.topology.residues:
                    if residue.name == 'HOH':
                        residue.name = 'SOL'

                print(f"Trajectory loaded using every {n}th step")
                savesubfolder = f'{savefolder}/{run_folder}'
                os.makedirs(savesubfolder, exist_ok = True)

                strided_traj.save(f'{savesubfolder}/traj_{idx}_stride_{stride}.gro')

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Convert TPS .dcd trajectories to .gro for GRADE.",
    )
    p.add_argument("--folder", required=True,
                   help="TPS output folder containing run_*/traj_*.dcd")
    p.add_argument("--savefolder", required=True,
                   help="Where to write the .gro trajectories")
    p.add_argument("--gro-folder", default="gro_files",
                   help="Folder containing conf.gro (default: gro_files)")
    p.add_argument("--burst", type=int, default=100000,
                   help="Steps between saved frames; must match the "
                        "sampler's save stride (default: 100000)")
    p.add_argument("--stride", type=int, default=10,
                   help="Subsample trajectory every N frames before "
                        "saving (default: 10)")
    p.add_argument("-n", type=int, default=1,
                   help="Load stride: read every N-th frame from the .dcd "
                        "(default: 1)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.folder, args.savefolder, args.gro_folder,
        burst=args.burst, stride=args.stride, n=args.n)
