import numpy as np  # Import NumPy for numerical operations and array handling


# Define a class representing a uniform selection probability
class UniformSelection(object):

    # Constructor does not require any parameters
    def __init__(self):

        pass  # No initialization needed for uniform selection

    # Compute the selection weight for each configuration
    def compute(self, current_x : np.ndarray):

        # Return an array of ones with length equal to the number of configurations
        # This means every configuration has equal probability (uniform selection)
        return np.ones((current_x.shape[0]))


# Define a class representing a Gaussian selection probability
class GaussianSelection(object):

    # Constructor initializes parameters for the Gaussian weighting
    def __init__(self, cv, k, cv_ref):

        self.cv = cv        # Collective variable used to evaluate configurations
        self.k = k          # Strength/width parameter controlling the Gaussian sharpness
        self.cv_ref = cv_ref  # Reference value of the collective variable (Gaussian center)

    # Compute the Gaussian selection weight for each configuration
    def compute(self, current_x : np.ndarray):

        # Evaluate the collective variable for the provided configurations
        cv_values = self.cv.compute(current_x)

        # Compute Gaussian weights based on the distance from the reference CV value
        # Configurations closer to cv_ref will have higher selection probability
        return np.exp(-self.k*(cv_values - self.cv_ref)**2)