import numpy as np  # Import NumPy for numerical operations and array handling

# Import the position update routine used to propagate the dynamics
from aaa_tps.miniMD.miniMD import update_positions


# Function that generates a trajectory starting from a shooting point
def generate_path(shooting_point : np.ndarray, 
                  force_function : callable,
                  max_path_length : int,
                  states : list,
                  configuration_output_frequency : int,
                  beta : float, 
                  timestep : float, 
                  diffusion_coefficient : float):
    """
    Returns a trajectory started from the shooting point and that ends when a state is reached.

    Parameters
    ----------
    force_function: callable
        Function that computes the force for the specific system simulated

    shooting_point : np.ndarray
        Initial configuration of the trajectory.

    max_path_length : int
        MAXIMUM number of integration steps of the trajectory
        
    states : list
        List of state indicator functions to check against if current_x is in a state (for use [state_A_indicator, state_B_indicator])
        
    configuration_output_frequency : int
        Output frequency
        
    beta : float
        1 / kT

    timestep : float
        Timestep of integration

    diffusion_coefficient : float
        Diffusion coefficient of the overdamped dynamics

    Returns
    -------
    trajectory : np.ndarray
        Trajectory started from shooting_point as initial configuration
    """

    # Copy the shooting point so the original configuration is not modified
    previous_x = shooting_point.copy()

    # Initialize the trajectory list with the starting configuration
    trajectory = [previous_x]

    # Counter for how many times the force function has been evaluated
    force_eval = 0

    # Flag that indicates whether a state has been reached
    state_reached = False

    # Check if the shooting point already lies inside one of the states
    for state in states:
        if state.is_in_state(previous_x): 
            state_reached = True

    # Only propagate the trajectory if the starting configuration is not already in a state
    if not state_reached:

        # Integrate the dynamics for at most max_path_length steps
        for step in range(max_path_length):

            # Compute the force acting on the current configuration
            current_force = force_function(previous_x)

            # Increment the force evaluation counter
            force_eval += 1

            # Propagate the system using overdamped Langevin dynamics
            previous_x = update_positions(previous_x, current_force, beta, timestep, diffusion_coefficient)

            # Save configurations according to the output frequency
            if configuration_output_frequency > 0:

                # Only store the configuration every configuration_output_frequency steps
                if step % configuration_output_frequency == 0:

                    # Append the new configuration to the trajectory
                    trajectory.append(previous_x)

                    # Check whether the new configuration lies inside any of the defined states
                    for state in states:
                        if state.is_in_state(previous_x): 
                            state_reached = True

            # If a state has been reached, stop the propagation early
            if state_reached: break

    # Convert the trajectory list to a NumPy array and return it together with the force evaluation count
    return np.array(trajectory), force_eval