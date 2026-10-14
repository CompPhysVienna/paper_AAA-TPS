import numpy as np  # Numerical library used for arrays
from tqdm import tqdm  # Progress bar for loops


# Function returning the index of the configuration of a path matching a given point
def _find_closest_index(path, point, tol):
    """
    Return the index of the configuration in `path` closest to `point`,
    or None if the distance is larger than `tol`.
    """

    # Distance of every configuration of the path from the point
    dists = np.linalg.norm(path - point, axis=1)

    # Index of the closest configuration
    idx = np.argmin(dists)

    # Return the index only if the match is within tolerance
    return idx if dists[idx] <= tol else None


# Function checking whether two paths are numerically identical
def _check_same_path(path1, path2):
    """
    Return True if the two paths are the same within a tight tolerance
    (i.e. the second one is the result of a rejected trial).
    """

    # Paths with different lengths cannot be the same
    if path1.shape != path2.shape:
        return False

    return np.allclose(path1, path2, rtol=1e-8, atol=1e-10)


# Function computing the overlap chains for one-way shooting based algorithms
def compute_overlap_chain_1w(paths, shooting_points, tol=1e-6):
    """
    Computes the fractional overlap between each reference trajectory and the
    subsequent trajectories of a one-way shooting transition path sampling chain
    (one-way shooting, ARA-TPS and AAA-TPS).

    Each trajectory in the chain is either a rejection (identical to the
    previous one) or an accepted move that shares with the previous trajectory
    the segment before (forward shot) or after (backward shot) the shooting point.

    Parameters
    ----------
    paths : list of np.ndarray of shape (N, 2)
        Trajectories sampled by the TPS chain.
    shooting_points : array-like of shape (num_paths, 2)
        Shooting point used to generate each trajectory (`shooting_points[i]`
        generated `paths[i]`).
    tol : float, optional
        Tolerance used to identify identical configurations.

    Returns
    -------
    overlaps : list of np.ndarray
        `overlaps[i][k]` is the fraction of configurations of `paths[i]` still
        contained in `paths[i+k]`. Each array stops at the first zero (full
        decorrelation) or at the end of the chain.
    """

    overlaps = []
    num_paths = len(paths)
    assert len(shooting_points) == num_paths, "Mismatch between shooting point list and path list"

    for i in tqdm(range(num_paths)):
        reference_path = paths[i]
        last_accepted_path = reference_path.copy()
        reference_length = len(reference_path)

        # Inclusive indexes delimiting the segment of the reference path still shared
        backward_idx = 0
        forward_idx = reference_length - 1

        fractions = [1.0]  # Path i compared with itself

        for j in range(i + 1, num_paths):
            current_path = paths[j]
            previous_path = paths[j - 1]

            # A rejected trial leaves the overlap with the reference unchanged
            rejected = _check_same_path(current_path, previous_path)
            if rejected:
                fractions.append(fractions[-1])
                continue

            current_shooting_point = shooting_points[j]
            test_reference_idx = _find_closest_index(reference_path, current_shooting_point, tol)

            lost_overlap = False

            if test_reference_idx is not None:
                # The shooting point lies on the reference path: shrink the shared window
                is_forward_shot = np.linalg.norm(last_accepted_path[0] - current_path[0], axis=-1) <= tol
                is_backward_shot = np.linalg.norm(last_accepted_path[-1] - current_path[-1], axis=-1) <= tol

                if is_forward_shot:
                    forward_idx = min(forward_idx, test_reference_idx)
                elif is_backward_shot:
                    backward_idx = max(backward_idx, test_reference_idx)
                else:
                    # The accepted path shares no endpoint with its parent
                    lost_overlap = True
            else:
                # The shooting point is outside the shared window: check that
                # the window boundaries are still contained in the current path
                backward_idx_on_current = _find_closest_index(current_path, reference_path[backward_idx], tol)
                forward_idx_on_current = _find_closest_index(current_path, reference_path[forward_idx], tol)
                if backward_idx_on_current is None and forward_idx_on_current is None:
                    lost_overlap = True

            if lost_overlap or backward_idx > forward_idx:
                fractions.append(0.0)
                break

            # Number of shared configurations (inclusive indexes) over reference length
            fractions.append((forward_idx - backward_idx + 1) / reference_length)

            last_accepted_path = current_path.copy()

        overlaps.append(np.array(fractions))

    return overlaps


# Function computing the overlap chains for two-way shooting
def compute_overlap_chain_2w(paths, shooting_points, tol=1e-6):
    """
    Computes the fractional overlap between each reference trajectory and the
    subsequent trajectories of a two-way shooting transition path sampling chain.

    In two-way shooting an accepted trajectory shares only the shooting point
    with its parent, so the overlap with a reference path is 1/L as long as the
    new shooting points lie on the reference path, and zero afterwards.

    Parameters and returns are the same as for `compute_overlap_chain_1w`.
    """

    overlaps = []
    num_paths = len(paths)
    assert len(shooting_points) == num_paths, "Mismatch between shooting point list and path list"

    for i in tqdm(range(num_paths)):
        reference_path = paths[i]
        reference_length = len(reference_path)

        fractions = [1.0]  # Path i compared with itself
        last_frac = 1.0

        for j in range(i + 1, num_paths):
            current_path = paths[j]
            previous_path = paths[j - 1]

            # A rejected trial leaves the overlap with the reference unchanged
            if _check_same_path(current_path, previous_path):
                fractions.append(last_frac)
                continue

            if _find_closest_index(reference_path, shooting_points[j], tol) is not None:
                last_frac = 1 / reference_length
                fractions.append(last_frac)
            else:
                fractions.append(0.0)
                break

        overlaps.append(np.array(fractions))

    return overlaps


# Function fusing the replica and path axes of a list of overlap chains
def flatten_overlaps(overlaps_by_replica):
    """
    Flatten overlaps_by_replica[replica][path][k] into flattened[path][k].
    """
    return [overlap_chain for replica in overlaps_by_replica for overlap_chain in replica]


# Function averaging the overlap chains as a function of the force evaluations
def average_overlap_vs_force_evals(overlap_list, force_evals, num_grid_points=200,
                                   x_grid=None, spacing="linear", x_min=None):
    """
    Average path overlap as a function of the cumulative number of force
    evaluations spent after the reference path.

    Parameters
    ----------
    overlap_list : list of array-like
        overlap_list[i][k] = fractional overlap between paths[i] and paths[i+k].
        Short rows are continued with their last value.
    force_evals : array-like of shape (num_paths,)
        force_evals[j] = force evaluations spent to generate path j.
    num_grid_points : int, optional
        Number of points of the force-evaluation grid.
    x_grid : array-like, optional
        Explicit force-evaluation grid (overrides spacing, x_min and num_grid_points).
    spacing : {'linear', 'log'}, optional
        Grid spacing. With 'log' the grid contains x=0 followed by
        num_grid_points-1 log-spaced points between x_min and the total budget.
    x_min : float, optional
        Smallest nonzero grid point for 'log' spacing (default: cheapest trial).

    Returns
    -------
    x_grid : np.ndarray
        Force-evaluation grid.
    mean : np.ndarray
        Average overlap on the grid (NaN where no reference path reaches).
    counts : np.ndarray
        Number of reference paths contributing to each grid point.
    """

    num_paths = len(overlap_list)
    force_evals = np.asarray(force_evals, dtype=float)
    assert len(force_evals) == num_paths, "force_evals must have one entry per path"

    # Cumulative force evaluations through each path
    C = np.cumsum(force_evals)

    if x_grid is None:
        x_max_global = C[-1] - C[0]
        if spacing == "linear":
            x_grid = np.linspace(0.0, x_max_global, num_grid_points)
        elif spacing == "log":
            if num_grid_points < 2:
                raise ValueError("num_grid_points must be >= 2 for log spacing")
            if x_min is None:
                positive_fe = force_evals[force_evals > 0]
                x_min = positive_fe.min() if positive_fe.size > 0 else x_max_global / num_grid_points
            if x_min <= 0 or x_min >= x_max_global:
                raise ValueError(f"x_min ({x_min}) must be in (0, x_max_global={x_max_global})")
            log_part = np.logspace(np.log10(x_min), np.log10(x_max_global), num_grid_points - 1)
            x_grid = np.concatenate(([0.0], log_part))
        else:
            raise ValueError("spacing must be 'linear' or 'log'")
    else:
        x_grid = np.asarray(x_grid, dtype=float)

    n_grid = len(x_grid)
    sum_y = np.zeros(n_grid)
    counts = np.zeros(n_grid, dtype=int)

    for i, row in enumerate(overlap_list):
        row = np.asarray(row)
        x_max_i = C[-1] - C[i]                  # Force evaluations available after path i
        x_explicit = C[i:i + len(row)] - C[i]   # Force evaluations at each stored offset

        valid = x_grid <= x_max_i
        if not np.any(valid):
            continue
        g = x_grid[valid]

        # Right-continuous step lookup: value of the last stored offset before g
        k = np.searchsorted(x_explicit, g, side="right") - 1
        k = np.clip(k, 0, len(row) - 1)

        sum_y[valid] += row[k]
        counts[valid] += 1

    mean = np.full(n_grid, np.nan)
    has_data = counts > 0
    mean[has_data] = sum_y[has_data] / counts[has_data]

    return x_grid, mean, counts


# Function computing the average force evaluations needed for full path renewal
def average_force_evals_to_zero_overlap(overlap_list, force_evals):
    """
    Average number of force evaluations needed for the overlap with a reference
    path to reach zero (complete renewal of the trajectory).

    Reference paths that are still overlapping when the chain ends are
    right-censored and are reported separately, not included in the mean.

    Parameters
    ----------
    overlap_list : list of array-like
        Overlap chains as returned by compute_overlap_chain_1w/2w.
    force_evals : array-like of shape (num_paths,)
        force_evals[j] = force evaluations spent to generate path j.

    Returns
    -------
    result : dict
        mean_fe_to_zero, std_fe_to_zero, n_decayed, n_censored, n_no_data,
        decay_fe_values, censored_fe_lower_bounds.
    """

    num_paths = len(overlap_list)
    force_evals = np.asarray(force_evals, dtype=float)
    assert len(force_evals) == num_paths, "force_evals must have one entry per path"

    C = np.cumsum(force_evals)

    decay_fe = []
    censored_fe = []
    n_no_data = 0

    for i, row in enumerate(overlap_list):
        row = np.asarray(row)
        if len(row) < 2:
            # Last path of the chain: no successor to compare with
            n_no_data += 1
            continue

        k_last = len(row) - 1
        fe_last = C[i + k_last] - C[i]

        if row[-1] == 0.0:
            decay_fe.append(fe_last)
        else:
            censored_fe.append(fe_last)

    decay_fe = np.asarray(decay_fe)
    censored_fe = np.asarray(censored_fe)
    n_decayed = len(decay_fe)

    return {
        "mean_fe_to_zero": float(decay_fe.mean()) if n_decayed > 0 else np.nan,
        "std_fe_to_zero": float(decay_fe.std(ddof=1)) if n_decayed > 1 else np.nan,
        "n_decayed": n_decayed,
        "n_censored": len(censored_fe),
        "n_no_data": n_no_data,
        "decay_fe_values": decay_fe,
        "censored_fe_lower_bounds": censored_fe,
    }
