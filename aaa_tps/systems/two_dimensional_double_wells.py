import numpy as np  # Import NumPy for numerical operations and array handling
from numba import jit  # Import Numba JIT compiler to accelerate numerical functions


# Define a class representing a standard asymmetric double-well potential
class StandardDoubleWell(object):

    # Constructor initializes the parameters controlling the potential
    def __init__(self, barrier = 3, asymmetry = 0):

        self.barrier = barrier  # Controls the height of the potential barrier
        self.asymmetry = asymmetry  # Introduces asymmetry between the wells
    
    # Return system parameters as a NumPy array for use in JIT-compiled functions
    def get_system_parameters(self):
        
        return np.array([self.barrier, self.asymmetry])
    
    # Public interface for computing the energy of a configuration
    def energy_function(self, current_x):

        # Call the static JIT-compiled energy function with the system parameters
        return self._energy_function(current_x, self.get_system_parameters())
    
    # Static method so it can be compiled efficiently with Numba
    @staticmethod
    @jit(nopython=True) 
    def _energy_function(current_x : np.ndarray, system_parameters : np.ndarray) -> float:
        """
        Calculates the potential energy given a configuration current_x. 
        
        Parameters
        ----------
        current_x : np.ndarray
            Current configuration to be propagated. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        Returns
        -------
        U : float
            Potential energy of the configuration

        """

        # Compute the potential energy of the configuration
        # The energy contains:
        # - a quartic double-well term (x^2 - 1)^2
        # - a coupling term between x and y
        # - an asymmetry term that tilts the potential
        return system_parameters[0] * ((current_x[0]**2 - 1)**2 + (current_x[0] - current_x[1])**2 + system_parameters[1] * (current_x[0] + current_x[1]))

    
    # Public interface for computing the force acting on a configuration
    def force_function(self, current_x):

        # Call the static JIT-compiled force function with the system parameters
        return self._force_function(current_x, self.get_system_parameters())
    

    # Static method compiled with Numba for efficient force evaluation
    @staticmethod
    @jit(nopython=True) 
    def _force_function(current_x : np.ndarray, system_parameters : np.ndarray) -> np.ndarray:
        """
        Calculates the force given a configuration current_x. 

        Parameters
        ----------
        current_x : np.ndarray
            Current configuration to be propagated. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        Returns
        -------
        force : np.ndarray
            Force corresponding the provided configuration.

        """

        # Initialize the force vector (2D system)
        force = np.zeros(2)
        
        # Force components correspond to the negative gradient of the potential energy
        # First component (x-direction)
        force[0] = -2 * system_parameters[0] * (2 * current_x[0]**3 -current_x[0] - current_x[1] + 0.5 * system_parameters[1])

        # Second component (y-direction)
        force[1] = 2 * system_parameters[0] * (current_x[0] - current_x[1] - 0.5 * system_parameters[1])

        # Return the computed force vector
        return force


# Define a class representing an alternative bistable double-well potential
class BistableDoubleWell(object):

    # Constructor initializes the parameters controlling the potential shape
    def __init__(self, A = 3, B = 0):

        self.A = A  # Controls overall scaling of the potential
        self.B = B  # Controls the strength of the radial quartic term
    

    # Return system parameters as a NumPy array for use in JIT-compiled functions
    def get_system_parameters(self):
        
        return np.array([self.A, self.B])


    # Public interface for computing the potential energy
    def energy_function(self, current_x):

        # Call the static JIT-compiled energy function
        return self._energy_function(current_x, self.get_system_parameters())

    # Static method compiled with Numba for efficient evaluation
    @staticmethod
    @jit(nopython=True) 
    def _energy_function(current_x : np.ndarray, system_parameters : np.ndarray) -> float:
        """
        Calculates the potential energy given a configuration current_x. 
        
        Parameters
        ----------
        current_x : np.ndarray
            Current configuration to be propagated. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        Returns
        -------
        U : float
            Potential energy of the configuration

        """

        # Compute the bistable potential energy
        # This potential depends on the radial distance from the origin
        # and contains a quartic term creating a ring-like bistable structure
        return system_parameters[0]/8 * (system_parameters[1] * (current_x[0]**2  + current_x[1]**2 - 4)**2  + current_x[1]**2)


    # Public interface for computing the force acting on a configuration
    def force_function(self, current_x):

        # Call the static JIT-compiled force function
        return self._force_function(current_x, self.get_system_parameters())

    # Static method compiled with Numba for efficient force evaluation
    @staticmethod
    @jit(nopython=True) 
    def _force_function(current_x : np.ndarray, system_parameters : np.ndarray) -> np.ndarray:
        """
        Calculates the force given a configuration current_x. 

        Parameters
        ----------
        current_x : np.ndarray
            Current configuration to be propagated. The shape of the array(current_x.shape) can vary depending on the system which is simulated.

        Returns
        -------
        force : np.ndarray
            Force corresponding the provided configuration.

        """

        # Initialize the force vector for the 2D system
        force = np.zeros(2)
        
        # Force components correspond to the negative gradient of the potential
        # x-direction component
        force[0] = -system_parameters[0]*system_parameters[1]/2 * current_x[0] * (current_x[0]**2 + current_x[1]**2 - 4)

        # y-direction component
        force[1] = -system_parameters[0]/2*current_x[1] * (system_parameters[1] * (current_x[0]**2 + current_x[1]**2 - 4) + 0.5)

        # Return the computed force vector
        return force