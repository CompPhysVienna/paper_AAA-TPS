import numpy as np  # Import NumPy for numerical operations and array handling

# Define a class representing a linear collective variable (CV)
class linear_cv(object):

    # Constructor: initializes coefficients for the linear combination
    def __init__(self, a=1, b=1):

        self.a = a  # Weight applied to the first coordinate (x component)
        self.b = b  # Weight applied to the second coordinate (y component)

    # Compute the value of the collective variable for a given configuration
    def compute(self, current_x : np.ndarray) -> float:
        """
        Calculates the collective variable zeta given a configuration current_x.

        Parameters
        ----------
        current_x : np.ndarray
            Current configuration. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        Returns
        -------
        zeta : float
            CV corresponding the provided configuration.
        """

        # Return the linear combination of the first two coordinates of the configuration
        # The ellipsis (...) allows the method to work with arrays of arbitrary leading dimensions
        return self.a*current_x[..., 0] + self.b*current_x[..., 1]
    

# Define a class representing a circular/elliptical collective variable
class circular_cv(object):

    # Constructor: defines ellipse axes and rotation angle
    def __init__(self, axes: tuple = (1.0, 1.0), angle: float = 0.0):

        self.a, self.b = axes  # Semi-axis lengths of the ellipse (x and y directions)

        # Precompute cosine and sine of the rotation angle for efficiency
        self.cos_t = np.cos(angle)
        self.sin_t = np.sin(angle)

    # Compute the value of the collective variable for a given configuration
    def compute(self, current_x : np.ndarray, center: tuple = (0.0, 0.0)) -> float:
        """
        Calculates the collective variable zeta given a configuration current_x.

        Parameters
        ----------
        current_x : np.ndarray
            Current configuration. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        Returns
        -------
        zeta : float
            CV corresponding the provided configuration.
        """

        # Shift the coordinates so that the ellipse is centered at the specified center
        x_shifted = current_x[..., 0] - center[0]
        y_shifted = current_x[..., 1] - center[1]

        # Rotate coordinates according to the specified angle
        # This aligns the coordinate system with the rotated ellipse
        x_rot = self.cos_t * x_shifted + self.sin_t * y_shifted
        y_rot = -self.sin_t * x_shifted + self.cos_t * y_shifted

        # Compute the scaled squared distance in the rotated coordinate system
        # This corresponds to the equation of an ellipse: (x/a)^2 + (y/b)^2
        zeta = (x_rot / self.a)**2 + (y_rot / self.b)**2

        # Return the computed collective variable value
        return zeta