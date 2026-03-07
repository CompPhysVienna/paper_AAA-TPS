import numpy as np  # Import NumPy for numerical operations and array handling

# Import collective variable (CV) definitions used to evaluate configurations
from aaa_tps.cvs.cvs import linear_cv, circular_cv


# Define a state based on a linear collective variable and bounds
class linear_state(object):

    # Constructor initializes the bounds that define the state
    def __init__(self, bounds : tuple = (-np.inf, np.inf)):
        
        self.bounds = bounds  # Tuple containing lower and upper bounds of the state
        self.cv = linear_cv()  # Create an instance of the linear collective variable

    # Determine whether the provided configuration lies inside the state
    def is_in_state(self, current_x : np.ndarray) -> bool:
    
        """
        Returns a bool or bool array which is true if current_x is within in the bounds.

        Parameters
        ----------
        current_x : np.ndarray
            Current configuration. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        cv_function : callable
            Function that accepts the current configuration and maps it onto the collective variable.
            
        bounds : (float, float)
            The lower and upper bounds for x to be considered within the stable state


        Returns
        -------
        inState : bool
            True for each x in current_x if x is within the bounds
        """

        # Compute the collective variable value(s) for the provided configuration
        cv_values = self.cv.compute(current_x)
        
        # Check whether the CV values lie within the specified bounds
        # Returns a boolean (or boolean array) indicating membership in the state
        return (cv_values > self.bounds[0]) & (cv_values < self.bounds[1])


# Define a state based on a circular/elliptical collective variable
class circular_state(object):

    # Constructor initializes geometric parameters describing the state
    def __init__(self, center : tuple = (0, 0), axes :  tuple = (1, 1), angle : float = 0, radius : float = 1.):

        self.center = center  # Center of the circular/elliptical region
        self.radius = radius  # Threshold value defining the boundary of the state
        self.cv = circular_cv(axes=axes, angle=angle)  # Create the circular/elliptical CV instance

    # Determine whether the provided configuration lies inside the state
    def is_in_state(self, current_x : np.ndarray) -> bool:
    
        """
        Returns a bool or bool array which is true if current_x is within in the bounds.

        Parameters
        ----------
        current_x : np.ndarray
            Current configuration. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        cv_function : callable
            Function that accepts the current configuration and maps it onto the collective variable.
            
        bounds : (float, float)
            The lower and upper bounds for x to be considered within the stable state


        Returns
        -------
        inState : bool
            True for each x in current_x if x is within the bounds
        """

        # Compute the collective variable value(s) for the configuration
        # The center is passed so the CV is evaluated relative to the desired origin
        cv_values = self.cv.compute(current_x, self.center)
        
        # Check whether the CV value lies within the radius threshold
        # This effectively tests whether the configuration is inside the circular/elliptical region
        return cv_values < self.radius