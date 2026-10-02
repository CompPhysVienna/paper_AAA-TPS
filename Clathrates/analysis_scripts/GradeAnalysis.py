import argparse
import fnmatch
import os
import subprocess


def run(savefolder, grade_binary="GRADE_updated/GRADE", stride=10):
    """
    Run GRADE on every .gro trajectory under savefolder/run_*/.

    Parameters
    ----------
    savefolder : str
        Folder written by DCD2GRO.py, containing run_*/traj_*_stride_*.gro.
    grade_binary : str
        Path to the GRADE executable.
    stride : int
        Frame stride used by DCD2GRO.py when writing the .gro files.
        Only used to build the filename pattern.
    """
    if not os.path.isfile(grade_binary):
        raise FileNotFoundError(
            f"GRADE binary not found at {grade_binary}. "
            "Build it first (cd GRADE_updated && make), or pass "
            "--grade-binary /path/to/GRADE."
        )

    for run_folder in os.listdir(savefolder):
        run_folder_path = os.path.join(savefolder,run_folder)

        # Skip non-directories and hidden folders
        if not os.path.isdir(run_folder_path) or run_folder.startswith('.'):
            continue
        
        # Process trajectory files
        for gro_file in os.listdir(run_folder_path):
            pattern = f'traj_*_stride_{stride}*.gro'
            print(f'checking for pattern {pattern}')
            if fnmatch.fnmatch(gro_file, pattern):
                print(f"Matched: {gro_file}")
                gro_file_path = os.path.join(run_folder_path, gro_file)
                print(f'Run on {gro_file_path}')

                command = [grade_binary, "-i", gro_file_path]
                
                try:
                    result = subprocess.run(
                        command,
                        check=True,          # Raises error if command fails
                        capture_output=True, # Captures stdout/stderr
                        text=True           # Returns output as string (not bytes)
                    )
                    print(f"Output for {gro_file}:")
                    print(result.stdout)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {gro_file}:")
                    print(e.stderr)

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run GRADE cage analysis on DCD2GRO output.",
    )
    p.add_argument("--savefolder", required=True,
                   help="Folder written by DCD2GRO.py, containing "
                        "run_*/traj_*_stride_*.gro")
    p.add_argument("--grade-binary", default="GRADE_updated/GRADE",
                   help="Path to the GRADE executable "
                        "(default: GRADE_updated/GRADE)")
    p.add_argument("--stride", type=int, default=10,
                   help="Stride used by DCD2GRO.py; must match "
                        "(default: 10)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.savefolder,
        grade_binary=args.grade_binary,
        stride=args.stride)