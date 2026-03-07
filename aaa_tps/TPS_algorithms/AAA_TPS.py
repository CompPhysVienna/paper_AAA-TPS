import numpy as np  # Numerical library used for arrays and random sampling
from functools import partial  # Allows fixing arguments of a function
from tqdm import tqdm  # Progress bar for loops
import os  # File and directory operations
import pickle  # Used to serialize Python objects (saving trajectories)
import gc  # Garbage collector for freeing memory


# Class implementing AAA-TPS (Always-Accepting Algorithm for TPS)
class AAA_TPS(object):

    # Constructor initializing the TPS simulation
    def __init__(self,
                 max_path_length, 
                 state_A,
                 state_B,
                 generate_path,
                 p_sel,
                 init_path,
                 force_A2B=True, 
                 output_dir="./output",
                 runname="AAA-TPS",
                 n_replicas=1,
                 symmetrize_histo=False,
                ):
        
        # Whether histogram output should be symmetrized
        self._symmetric = symmetrize_histo

        # Ensure number of replicas is valid
        assert n_replicas > 0, "Number of replicas must be bigger than 0"

        # Maximum allowed trajectory length
        self.max_path_length = max_path_length

        # Stable states defining the reactive transition
        self.state_A = state_A
        self.state_B = state_B
        
        # Ensure initial path is shorter than the maximum path length
        assert init_path.shape[0] < max_path_length, f"Initial path longer than the maxi path length: L = {init_path.shape[0]}, Lmax = {max_path_length}"

        # Check if path starts in A and ends in B
        starts_in_A = self.state_A.is_in_state(init_path[0])
        ends_in_B = self.state_B.is_in_state(init_path[-1])
        path_A2B = starts_in_A and ends_in_B

        # If not A→B, check if it is B→A
        if not path_A2B:
            starts_in_B = self.state_B.is_in_state(init_path[0])
            ends_in_A = self.state_A.is_in_state(init_path[-1])
            is_reactive_path = starts_in_B and ends_in_A

            # Ensure path is reactive
            assert is_reactive_path, "The initial path is not reactive"

            # Optionally reverse the path to enforce A→B orientation
            if force_A2B:
                init_path = init_path[::-1]

        # Ensure the path contains exactly one frame in each stable state
        in_A = np.array([state_A.is_in_state(x) for x in init_path])
        in_B = np.array([state_B.is_in_state(x) for x in init_path])

        count_A = np.count_nonzero(in_A)
        count_B = np.count_nonzero(in_B)

        assert count_A == 1 and count_B == 1, f"The initial path has more than one point in each state. In A: {count_A}. In B: {count_B}"

        # Store the initial trajectory
        self.init_path = init_path

        # Bind the state list to the path generation function
        self.generate_path = partial(generate_path,
                                     states=[state_A, state_B])
        
        # Shooting point selection probability object
        self.p_sel = p_sel

        # Create output directories if they do not exist
        if n_replicas == 1:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
        else:
            for replica in range(n_replicas):
                rep_output_dir = os.path.join(output_dir, f"R_{replica:04d}")
                if not os.path.exists(rep_output_dir):
                    os.makedirs(rep_output_dir)

        # Store directory paths and metadata
        self.output_dir = output_dir
        self.root_output_dir = output_dir
        self.runname = runname
        self.n_replicas = n_replicas

        # Initialize simulation state
        self._initialize()


    # Helper function to set the output directory
    def _set_output_dir(self, output_dir):

        self.output_dir = output_dir

        # Create directory if necessary
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return
    

    # Initialize simulation variables and output containers
    def _initialize(self):
        
        # Save the initial path
        np.savetxt(os.path.join(self.output_dir, f"init_path_{self.runname}.txt"), self.init_path)

        # Set current path to the initial path
        self.current_path = self.init_path.copy()

        # Flag indicating whether equilibration has finished
        self._equilibrated = False

        # Output containers
        self.trajectory_ensemble = []
        self.force_evaluations = []
        self.lengths = []

        # Weight statistics
        self.accepted_weights_lengths = []
        self.accepted_p_sel_lengths = []
        self.accepted_p_sel_norm_lengths = []
        self.accepted_weights_trajectory_ensemble = []
        self.accepted_p_sel_trajectory_ensemble = []
        self.accepted_p_sel_norm_trajectory_ensemble = []
        self.total_weights = []
        self.total_p_sel = []
        self.total_p_sel_norm = []

        # Shooting point statistics
        self.accepted_shooting_point_indexes = []
        self.accepted_shooting_points_bw = []
        self.accepted_shooting_points_fw = []
        self.total_shooting_points = []

        # TPS statistics counters
        self.reactive_trials = 0
        self.accepted_trials = 0
        self.rejected_d2lr_trials = 0
        self.total_trials = 0
        self.equilibration_trials = 0


    # Main TPS simulation loop
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

        # Configure replica output directory if necessary
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

        # Compute shooting point selection probabilities
        p_sel_current_path = self.p_sel.compute(self.current_path)
        p_sel_current_path_norm = p_sel_current_path.sum()

        # Normalize weights
        w = p_sel_current_path / p_sel_current_path_norm

        tqdm_title = "" if replica < 0 else f"Replica {replica+1:04d}"

        # Main TPS trial loop
        for trial in tqdm(range(n_trials), desc=tqdm_title, unit="trials", disable=(not show_progress)):

            # Select shooting point from current path
            shooting_point_index = np.random.choice(np.arange(0, self.current_path.shape[0]), p=w)
            shooting_point = self.current_path[shooting_point_index]

            if self._equilibrated:
                self.total_shooting_points.append(shooting_point)
                self.total_weights.append(w[shooting_point_index])
                self.total_p_sel.append(p_sel_current_path[shooting_point_index])
                self.total_p_sel_norm.append(p_sel_current_path_norm)

            # Split current path into backward and forward segments
            backward_trj = self.current_path[:shooting_point_index]
            forward_trj = self.current_path[shooting_point_index:]

            # Generate new half trajectory
            half_trj, force_eval = self.generate_path(
                shooting_point=shooting_point, 
                max_path_length=self.max_path_length - max(forward_trj.shape[0], (backward_trj.shape[0] + 1)), 
                configuration_output_frequency=configuration_output_frequency)

            # Check where the generated trajectory ends
            ends_in_A = self.state_A.is_in_state(half_trj[-1])
            ends_in_B = self.state_B.is_in_state(half_trj[-1])

            # If it reaches a stable state it is reactive
            if ends_in_A or ends_in_B:

                if self._equilibrated:
                    self.reactive_trials += 1
                    self.accepted_trials += 1
                
                # Construct new path
                if ends_in_B:
                    self.current_path = np.vstack([backward_trj, half_trj])

                    if self._equilibrated:
                        self.accepted_shooting_points_fw.append(shooting_point)
                        new_shooting_point_index = len(backward_trj)

                else:
                    self.current_path = np.vstack([half_trj[::-1], forward_trj[1:]])

                    if self._equilibrated:
                        self.accepted_shooting_points_bw.append(shooting_point)
                        new_shooting_point_index = len(half_trj) - 1

                # Update selection probabilities
                p_sel_current_path = self.p_sel.compute(self.current_path)
                p_sel_current_path_norm = p_sel_current_path.sum()
                w = p_sel_current_path / p_sel_current_path_norm
        
            # Store path statistics after equilibration
            if self._equilibrated:

                if path_length_output_frequency > 0 and trial % path_length_output_frequency == 0:
                    self.lengths.append(self.current_path.shape[0])
                    self.accepted_shooting_point_indexes.append(shooting_point_index)
                    self.accepted_weights_lengths.append(w[new_shooting_point_index])
                    self.accepted_p_sel_lengths.append(p_sel_current_path[new_shooting_point_index])
                    self.accepted_p_sel_norm_lengths.append(p_sel_current_path_norm)

                if force_evaluation_output_frequency > 0 and trial % force_evaluation_output_frequency == 0:
                    self.force_evaluations.append(force_eval)

                if path_output_frequency > 0 and trial % path_output_frequency == 0:
                    self.trajectory_ensemble.append(self.current_path)
                    self.accepted_weights_trajectory_ensemble.append(w[new_shooting_point_index])
                    self.accepted_p_sel_trajectory_ensemble.append(p_sel_current_path[new_shooting_point_index])
                    self.accepted_p_sel_norm_trajectory_ensemble.append(p_sel_current_path_norm)

                if dump_output_frequency > 0 and trial % dump_output_frequency == 0:
                    self.dump_output(trial)

            # Mark equilibration completion
            if trial == equilibration_trials:
                self._equilibrated = True
                self.equilibration_trials += equilibration_trials

        # Final output dump
        self.dump_output(n_trials, paths=dump_paths, histo=dump_histo, screen=(replica < 0), free_memory=False)

        self.total_trials += n_trials

        if replica >= 0:
            self.init_path = self.current_path

        return
    

    # Function writing simulation outputs to disk
    def dump_output(self, trial, paths=True, histo=True, screen=False, free_memory=False):
            
            total_sampling_trials = self.total_trials + trial - self.equilibration_trials

            # Save shooting point statistics
            np.savetxt(os.path.join(self.output_dir, f"tot_sp_{self.runname}.txt"), np.array(self.total_shooting_points))
            np.savetxt(os.path.join(self.output_dir, f"acc_sp_fw_{self.runname}.txt"), np.array(self.accepted_shooting_points_fw))
            np.savetxt(os.path.join(self.output_dir, f"acc_sp_bw_{self.runname}.txt"), np.array(self.accepted_shooting_points_bw))

            # Save weight statistics
            np.savetxt(os.path.join(self.output_dir, f"tot_w_{self.runname}.txt"), self.total_weights)
            np.savetxt(os.path.join(self.output_dir, f"tot_p_sel_{self.runname}.txt"), self.total_p_sel)
            np.savetxt(os.path.join(self.output_dir, f"tot_p_sel_norm_{self.runname}.txt"), self.total_p_sel_norm)

            np.savetxt(os.path.join(self.output_dir, f"acc_w_len_{self.runname}.txt"), self.accepted_weights_lengths)
            np.savetxt(os.path.join(self.output_dir, f"acc_p_sel_len_{self.runname}.txt"), self.accepted_p_sel_lengths)
            np.savetxt(os.path.join(self.output_dir, f"acc_p_sel_norm_len_{self.runname}.txt"), self.accepted_p_sel_norm_lengths)

            np.savetxt(os.path.join(self.output_dir, f"acc_w_te_{self.runname}.txt"), self.accepted_weights_trajectory_ensemble)
            np.savetxt(os.path.join(self.output_dir, f"acc_p_sel_te_{self.runname}.txt"), self.accepted_p_sel_trajectory_ensemble)
            np.savetxt(os.path.join(self.output_dir, f"acc_p_sel_norm_te_{self.runname}.txt"), self.accepted_p_sel_norm_trajectory_ensemble)

            # Save trajectory statistics
            np.savetxt(os.path.join(self.output_dir, f"lengths_{self.runname}.txt"), np.array(self.lengths))
            np.savetxt(os.path.join(self.output_dir, f"force_evaluations_{self.runname}.txt"), np.array(self.force_evaluations))
            
            # Save trajectory ensemble
            if paths:
                with open(os.path.join(self.output_dir, f"paths_{self.runname}.pkl"), 'wb') as f:
                    pickle.dump(self.trajectory_ensemble, f)

            # Compute weighted histogram of sampled configurations
            if histo:
                x_min, x_max = -2.75, 2.75
                y_min, y_max = -2.75, 2.75
                num_bins = 100

                # Compute trajectory weights (inverse normalization factors)
                weights_trajectory_ensemble = 1 / np.array(self.accepted_p_sel_norm_trajectory_ensemble)
                weights_norm_trajectory_ensemble = weights_trajectory_ensemble / weights_trajectory_ensemble.sum()

                # Stack all trajectory positions
                all_positions = np.vstack(self.trajectory_ensemble)

                # Number of frames per trajectory
                lengths = np.array([len(traj) for traj in self.trajectory_ensemble])

                # Assign weight per frame
                weights_per_position = np.repeat(weights_norm_trajectory_ensemble, lengths)

                # Optional symmetry reflection
                if self._symmetric:
                    reflected_positions = all_positions.copy()
                    reflected_positions[:, 1] *= -1

                    weights_per_position_reflected = weights_per_position.copy()

                    all_positions = np.vstack((all_positions, reflected_positions))
                    weights_per_position = np.hstack((weights_per_position, weights_per_position_reflected))

                    weights_per_position /= np.sum(weights_per_position)

                # Compute weighted 2D histogram
                Hist, xedges, yedges = np.histogram2d(
                    all_positions[:, 0],
                    all_positions[:, 1],
                    bins=num_bins,
                    range=[[x_min, x_max], [y_min, y_max]],
                    weights=weights_per_position,
                    density=True
                )

                np.savetxt(os.path.join(self.output_dir, f'P_x_TP_{self.runname}.txt'), Hist)

            # Save acceptance statistics
            with open(os.path.join(self.output_dir, f"acceptance_{self.runname}.txt"), 'w+') as f:
                f.write(f"Total accepted trials: {self.accepted_trials} -- Acceptance Ratio: {self.accepted_trials/total_sampling_trials}\n")
                f.write(f"Total reactive paths: {self.reactive_trials} -- Rejected Reactive paths due to length ratio: {self.rejected_d2lr_trials} --  Fraction rejected due to length ratio: {self.rejected_d2lr_trials/self.reactive_trials}\n")
            
            if screen:
                print()
                print(f"Total accepted trials: {self.accepted_trials} -- Acceptance Ratio: {self.accepted_trials/total_sampling_trials}")
                print(f"Total reactive paths: {self.reactive_trials} -- Rejected Reactive paths due to length ratio: {self.rejected_d2lr_trials} --  Fraction rejected due to length ratio: {self.rejected_d2lr_trials/self.reactive_trials}")

            # Save final path
            np.savetxt(os.path.join(self.output_dir, f"final_path_{self.runname}.txt"), self.current_path)
            np.savetxt(os.path.join(self.root_output_dir, f"final_path_{self.runname}.txt"), self.current_path)

            # Free memory if requested
            if free_memory:
                self.trajectory_ensemble.clear()
                del self.trajectory_ensemble
                gc.collect()