import numpy as np  # Numerical library used for arrays and random sampling
from functools import partial  # Used to pre-bind arguments to functions
from tqdm import tqdm  # Progress bar for loops
import os  # File and directory operations
import pickle  # Used to serialize Python objects (saving paths)
import gc  # Garbage collector for freeing memory


# Class implementing Two-Way Shooting Transition Path Sampling (TPS)
class TwoWayShootingTPS(object):

    # Constructor that initializes the TPS simulation
    def __init__(self,
                 max_path_length, 
                 state_A,
                 state_B,
                 generate_path,
                 p_sel,
                 init_path, 
                 output_dir="./output",
                 runname="2w",
                 n_replicas=1,
                 symmetrize_histo=False,
                 ):
        
        # Whether histogram output should be symmetrized
        self._symmetric = symmetrize_histo

        # Ensure at least one replica is used
        assert n_replicas > 0, "Number of replicas must be bigger than 0"

        # Maximum allowed trajectory length
        self.max_path_length = max_path_length

        # Stable states defining the reactive transition
        self.state_A = state_A
        self.state_B = state_B
        
        # Check init path is reactive and shorter than max length
        assert init_path.shape[0] < max_path_length, f"Initial path longer than the maxi path length: L = {init_path.shape[0]}, Lmax = {max_path_length}"

        # Determine whether the initial path starts/ends in the correct states
        starts_in_A = self.state_A.is_in_state(init_path[0])
        starts_in_B = self.state_B.is_in_state(init_path[0])
        ends_in_B = self.state_B.is_in_state(init_path[-1])
        ends_in_A = self.state_A.is_in_state(init_path[-1])

        # Check that the path connects the two states
        is_reactive_path = (starts_in_A and ends_in_B) or (starts_in_B and ends_in_A)
        assert is_reactive_path, "The initial path is not reactive"

        # Store the initial reactive trajectory
        self.init_path = init_path

        # Bind the states argument to the path generation function
        self.generate_path = partial(generate_path,
                                     states = [state_A, state_B])

        # Shooting point selection probability object
        self.p_sel = p_sel

        # Create output directories if they do not exist
        if n_replicas == 1:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            # For replica simulations create a directory per replica
            for replica in range(n_replicas):
                rep_output_dir = os.path.join(output_dir, f"R_{replica:04d}")
                if not os.path.exists(rep_output_dir):
                    os.makedirs(rep_output_dir)

        # Store directory paths and simulation metadata
        self.output_dir = output_dir
        self.root_output_dir = output_dir
        self.runname = runname
        self.n_replicas = n_replicas

        # Initialize internal variables and storage containers
        self._initialize()


    # Helper function to set the output directory
    def _set_output_dir(self, output_dir):

        self.output_dir = output_dir

        # Create directory if necessary
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return


    # Initialize simulation state and output containers
    def _initialize(self):
        
        # Save the initial path to disk
        np.savetxt(os.path.join(self.output_dir, f"init_path_{self.runname}.txt"), self.init_path)

        # Set current path to the initial path
        self.current_path = self.init_path.copy()

        # Flag indicating whether equilibration is finished
        self._equilibrated = False

        # Initialize output variables
        self.trajectory_ensemble = []  # Stores sampled trajectories
        self.force_evaluations = []  # Stores number of force evaluations
        self.lengths = []  # Stores trajectory lengths

        # Shooting point statistics
        self.accepted_shooting_point_indexes = []
        self.accepted_shooting_points = []
        self.total_shooting_points = []

        # TPS statistics counters
        self.reactive_trials = 0
        self.accepted_trials = 0
        self.rejected_d2lr_trials = 0
        self.total_trials = 0
        self.equilibration_trials = 0


    # Main function running the TPS simulation
    def run_tps_simulation(self, 
                 n_trials, 
                 configuration_output_frequency, 
                 path_length_output_frequency, 
                 path_output_frequency,
                 force_evaluation_output_frequency,
                 dump_output_frequency,
                 continue_simulation=False,
                 equilibration_trials=0,
                 show_progress=True,
                 dump_paths=True,
                 dump_histo=True,
                 replica=-1):
        
        # Configure replica output directory if running multiple replicas
        if replica >= 0:
            assert self.n_replicas > 1, "Set number of replicas in class initialization"
            self._set_output_dir(os.path.join(self.root_output_dir, f"R_{replica:04d}"))
        else:
            assert self.n_replicas == 1, "Specify number of simulation replica when running tps simulation"

        # Reset or continue simulation
        if not continue_simulation:
            self._initialize()
        else:
            self._equilibrated = False

        # Compute selection probabilities for the current path
        p_sel_current_path = self.p_sel.compute(self.current_path)
        p_sel_current_path_norm = p_sel_current_path.sum()

        # Normalize weights for selecting shooting points
        w = p_sel_current_path/p_sel_current_path_norm

        # Progress bar title
        tqdm_title = "" if replica < 0 else f"Replica {replica+1:04d}"

        # Main TPS trial loop
        for trial in tqdm(range(n_trials), desc = tqdm_title, unit="trials", disable=(not show_progress)):

            # Select shooting point from current path
            shooting_point_index = np.random.choice(np.arange(0, self.current_path.shape[0]), p=w)
            shooting_point = self.current_path[shooting_point_index]

            # Record shooting point if equilibration finished
            if self._equilibrated:
                self.total_shooting_points.append(shooting_point)

            # Generate forward trajectory from shooting point
            forward_trj, force_eval_fw = self.generate_path(shooting_point=shooting_point, 
                                                            max_path_length=self.max_path_length // 2 + 1, 
                                                            configuration_output_frequency=configuration_output_frequency)

            # Generate backward trajectory
            reverse_trj, force_eval_rv = self.generate_path(shooting_point=shooting_point, 
                                                            max_path_length=self.max_path_length // 2, 
                                                            configuration_output_frequency=configuration_output_frequency)

            # Check if endpoints fall in the correct states
            fw_in_A = self.state_A.is_in_state(forward_trj[-1])
            fw_in_B = self.state_B.is_in_state(forward_trj[-1])

            rv_in_A = self.state_A.is_in_state(reverse_trj[-1])
            rv_in_B = self.state_B.is_in_state(reverse_trj[-1])

            # Determine if the path connects A and B
            is_reactive_path = (fw_in_A and rv_in_B) or (fw_in_B and rv_in_A)

            # If path is reactive apply Metropolis acceptance
            if is_reactive_path:

                if self._equilibrated: 
                    self.reactive_trials +=1

                # Construct the proposed path
                proposed_path = np.vstack([reverse_trj[::-1], forward_trj[1:]])

                # Compute selection probability for the proposed path
                p_sel_proposed_path = self.p_sel.compute(proposed_path)
                p_sel_proposed_path_norm = p_sel_proposed_path.sum()

                # Acceptance probability based on selection weights
                p_sel_ratio =  p_sel_current_path_norm / p_sel_proposed_path_norm
                
                # Accept or reject according to Metropolis criterion
                if np.random.random() < p_sel_ratio:
                    self.current_path = proposed_path
                    p_sel_current_path = p_sel_proposed_path
                    p_sel_current_path_norm = p_sel_proposed_path_norm
                    w = p_sel_current_path/p_sel_current_path_norm

                    if self._equilibrated:
                        self.accepted_shooting_points.append(shooting_point)
                        self.accepted_trials += 1
                else:
                    if self._equilibrated:
                        self.rejected_d2lr_trials += 1

            # Store trajectory information after equilibration
            if self._equilibrated:

                if path_length_output_frequency > 0:
                    if trial % path_length_output_frequency == 0:
                        self.lengths.append(self.current_path.shape[0])

                if force_evaluation_output_frequency > 0:
                    if trial % force_evaluation_output_frequency == 0:
                        self.force_evaluations.append(force_eval_fw + force_eval_rv)

                if path_output_frequency > 0:
                    if trial % path_output_frequency == 0:
                        self.trajectory_ensemble.append(self.current_path)

                # Periodically dump outputs to disk
                if dump_output_frequency > 0:
                    if trial % dump_output_frequency == 0:
                        self.dump_output(trial)
            
            # Mark the end of equilibration
            if trial == equilibration_trials:
                self._equilibrated = True
                self.equilibration_trials += equilibration_trials

        # Final dump after simulation
        self.dump_output(n_trials, paths=dump_paths, histo=dump_histo, screen = (replica < 0), free_memory=False)

        # Update total number of trials performed
        self.total_trials += n_trials

        # Store last path as initial path for next replica stage
        if replica >=0:
            self.init_path = self.current_path

        return
    

    # Function writing simulation output to disk
    def dump_output(self, trial, paths=True, histo=True, screen=False, free_memory=False):
            
            # Total number of sampling trials excluding equilibration
            total_sampling_trials = self.total_trials + trial - self.equilibration_trials

            # Save shooting points
            np.savetxt(os.path.join(self.output_dir, f"tot_sp_{self.runname}.txt"), np.array(self.total_shooting_points))
            np.savetxt(os.path.join(self.output_dir, f"acc_sp_{self.runname}.txt"), np.array(self.accepted_shooting_points))

            # Save path statistics
            np.savetxt(os.path.join(self.output_dir, f"lengths_{self.runname}.txt"), np.array(self.lengths))
            np.savetxt(os.path.join(self.output_dir, f"force_evaluations_{self.runname}.txt"), np.array(self.force_evaluations))
            
            # Save trajectories using pickle
            if paths:
                with open(os.path.join(self.output_dir, f"paths_{self.runname}.pkl"), 'wb') as f:
                    pickle.dump(self.trajectory_ensemble, f)

            # Compute and save histogram of sampled configurations
            if histo:
                x_min, x_max = -2.75, 2.75
                y_min, y_max = -2.75, 2.75
                num_bins = 100

                # Combine all positions from sampled trajectories
                all_positions = np.vstack(self.trajectory_ensemble)

                # Optional symmetry operation for symmetric potentials
                if self._symmetric:
                    reflected_positions = all_positions.copy()
                    reflected_positions[:, 1] *= -1
                    all_positions = np.vstack((all_positions, reflected_positions))

                # Compute 2D probability histogram
                Hist, xedges, yedges = np.histogram2d(
                    all_positions[:, 0],
                    all_positions[:, 1],
                    bins=num_bins,
                    range=[[x_min, x_max], [y_min, y_max]],
                    density=True
                )

                # Save histogram
                np.savetxt(os.path.join(self.output_dir, f'P_x_TP_{self.runname}.txt'), Hist)

            # Save acceptance statistics
            with open(os.path.join(self.output_dir, f"acceptance_{self.runname}.txt"), 'w+') as f:
                f.write(f"Total accepted trials: {self.accepted_trials} -- Acceptance Ratio: {self.accepted_trials/total_sampling_trials}\n")
                f.write(f"Total reactive paths: {self.reactive_trials} -- Rejected Reactive paths due to length ratio: {self.rejected_d2lr_trials} --  Fraction rejected due to length ratio: {self.rejected_d2lr_trials/self.reactive_trials}\n")
            
            # Optionally print statistics to screen
            if screen:
                print()
                print(f"Total accepted trials: {self.accepted_trials} -- Acceptance Ratio: {self.accepted_trials/total_sampling_trials}")
                print(f"Total reactive paths: {self.reactive_trials} -- Rejected Reactive paths due to length ratio: {self.rejected_d2lr_trials} --  Fraction rejected due to length ratio: {self.rejected_d2lr_trials/self.reactive_trials}")
            
            # Save final trajectory
            np.savetxt(os.path.join(self.output_dir, f"final_path_{self.runname}.txt"), self.current_path)
            np.savetxt(os.path.join(self.root_output_dir, f"final_path_{self.runname}.txt"), self.current_path)
            
            # Optionally free memory from stored trajectories
            if free_memory:
                self.trajectory_ensemble.clear()
                del self.trajectory_ensemble
                gc.collect()