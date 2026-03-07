import numpy as np  # Import NumPy for numerical operations and array handling


# Function to compute the cumulative weighted average of a sequence
def cumulative_weighted_average(values, weights=None):
    """
    Compute the cumulative weighted average of a 1D array.

    Parameters:
        values (array-like): Data values (length N)
        weights (array-like): Corresponding weights (length N)

    Returns:
        np.ndarray: Cumulative weighted averages (length N)
    """

    # Convert input values to a NumPy array of floats
    values = np.asarray(values, dtype=float)

    # If weights are provided, convert them to a NumPy array
    if weights is not None:
        weights = np.asarray(weights, dtype=float)

        # Ensure that values and weights have the same length
        if len(values) != len(weights):
            raise ValueError("Values and weights must be the same length")
    else:
        # If no weights are provided, assume uniform weights
        weights = np.ones(values.shape)

    # Compute cumulative sum of the weights
    cum_weights = np.cumsum(weights)

    # Compute cumulative sum of weighted values
    cum_weighted_values = np.cumsum(values * weights)

    # Return the cumulative weighted average at each index
    return cum_weighted_values / cum_weights


# Function to compute the weighted autocorrelation function of a signal
def get_weighted_autocorrelation_function(signal: np.ndarray,
                                          weights: np.ndarray = None,
                                          norm: bool = True,
                                          mean: bool = True) -> np.ndarray:
    """
    Computes the weighted autocorrelation function of a signal.

    Parameters
    ----------
    signal : np.ndarray
        Statistical signal to be autocorrelated.

    weights : np.ndarray
        Array of weights associated with each element in the signal.

    norm : bool
        Whether to compute normalized or unnormalized autocorrelation function.
        Default True.

    mean : bool
        Whether to subtract the weighted mean from the signal.
        Default True.

    Returns
    -------
    autocorrelation_function : np.ndarray
        Weighted autocorrelation function of the signal.
    """

    # Convert signal to a NumPy array
    signal = np.asarray(signal)

    # Length of the signal
    M = len(signal)

    # If no weights are provided, use uniform weights
    if weights is None:
        weights = np.ones(M)
    else:
        # Convert weights to NumPy array
        weights = np.asarray(weights, dtype=float)

        # Ensure signal and weights have the same length
        if len(signal) != len(weights):
            raise ValueError("Values and weights must be the same length")

    # Optionally subtract the weighted mean from the signal
    if mean:

        # Compute the weighted mean of the signal
        w_mean = np.sum(weights * signal) / np.sum(weights)

        # Center the signal by subtracting the weighted mean
        signal = signal - w_mean

    # Initialize the array that will hold the autocorrelation values
    autocorrelation_function = np.zeros(M)

    # Compute the weighted autocorrelation for each lag
    for lag in range(M):

        # Compute combined weights for pairs separated by the current lag
        w = weights[:M - lag] * weights[lag:]

        # Select overlapping parts of the signal for the current lag
        x = signal[:M - lag]
        y = signal[lag:]

        # Compute the weighted correlation for this lag
        autocorrelation_function[lag] = np.sum(w * x * y) / np.sum(w)

    # Normalize the autocorrelation function if requested
    if norm:

        # The variance corresponds to the autocorrelation at lag 0
        var = autocorrelation_function[0]

        # Normalize all lags by the variance
        autocorrelation_function /= var

    # Return the weighted autocorrelation function
    return autocorrelation_function


# Function to compute the relative effective sample size given a set of weights
def relative_effective_sample_size(weights):
    """
    Compute the Effective Sample Size (ESS) given an array of weights.
    
    Parameters
    ----------
    weights : array-like
        An array of non-negative weights (e.g., from importance sampling or resampling).
    
    Returns
    -------
    float
        The effective sample size (ESS).
        
    Notes
    -----
    The ESS is defined as:
        ESS = (sum(w_i))^2 / sum(w_i^2)
    """

    # Convert the input weights to a NumPy array of floats
    weights = np.asarray(weights, dtype=float)

    # Total number of weights (sample size)
    N = weights.shape[0]

    # Ensure that all weights are non-negative
    # Negative weights would invalidate the ESS calculation
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative.")
    
    # Compute the sum of the weights
    sum_w = np.sum(weights)

    # Compute the sum of squared weights
    # This quantity determines how uneven the weights are
    sum_w_sq = np.sum(weights ** 2)
    
    # If the squared sum is zero (all weights are zero),
    # return 0 to avoid division by zero
    if sum_w_sq == 0:
        return 0.0
    
    # Compute the relative Effective Sample Size (ESS)
    # The formula (sum_w^2 / sum_w_sq) gives the standard ESS,
    # and dividing by N returns the ESS normalized by the sample size
    ess = (sum_w ** 2) / sum_w_sq / N

    # Return the normalized ESS
    return ess


def padded_chunk_cumsum(arr, n_sum):
    # Convert the input array to a NumPy array to ensure consistent behavior
    arr = np.asarray(arr)

    # Compute how many elements remain if the array length is not divisible by n_sum
    remainder = len(arr) % n_sum

    # Pad with zeros if needed so the array length becomes a multiple of n_sum
    if remainder != 0:
        pad_width = n_sum - remainder  # Number of zeros to add
        arr = np.pad(arr, (0, pad_width), mode='constant')  # Pad at the end of the array

    # Reshape the array into chunks of size n_sum
    # Each row now represents one chunk
    # Then compute the sum of each chunk
    chunk_sums = arr.reshape(-1, n_sum).sum(axis=1)

    # Compute and return the cumulative sum of the chunk sums
    return np.cumsum(chunk_sums)
