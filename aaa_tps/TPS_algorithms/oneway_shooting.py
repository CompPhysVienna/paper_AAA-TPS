import numpy as np  # Numerical library used for arrays and random sampling
from functools import partial  # Used to pre-bind arguments to functions
from tqdm import tqdm  # Progress bar for loops
import os  # File and directory operations
import pickle  # Used to serialize Python objects (saving paths)
import gc  # Garbage collector for freeing memory


# Class implementing One-Way Shooting Transition Path Sampling (TPS)
class OneWayShootingTPS(object):

    # Constructor that initializes the TPS simulation
    def __init__(self,
                 max_path_length, 
                 state_A,
                 state_B,
                 generate_path,
                 p_sel,
                 init_path, 
                 force_A2B=True,
                 output_dir="./output",
                 runname="1w",
                 n_replicas=1,
                 always_accept=False,
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

        # Check whether the path goes from A → B
        starts_in_A = self.state_A.is_in_state(init_path[0])
        ends_in_B = self.state_B.is_in_state(init_path[-1])
        path_A2B = starts_in_A and ends_in_B

        # If the path is not A → B, check if it is B → A
        if not path_A2B:
            starts_in_B = self.state_B.is_in_state(init_path[0])
            ends_in_A = self.state_A.is_in_state(init_path[-1])
            is_reactive_path = starts_in_B and ends_in_A

            # Ensure the path is reactive
            assert is_reactive_path, "The initial path is not reactive"

            # Optionally reverse path so simulation always runs A → B
            if force_A2B: 
                init_path = init_path[::-1]

        # Count how many frames lie in each state
        in_A = np.array([state_A.is_in_state(x) for x in init_path])
        in_B = np.array([state_B.is_in_state(x) for x in init_path])

        count_A = np.count_nonzero(in_A)
        count_B = np.count_nonzero(in_B)

        # Ensure only one frame lies in each stable state
        assert count_A == 1 and count_B == 1, f"The initial path has more than one point in each state. In A: {count_A}. In B: {count_B}"

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
            for replica in range(n_replicas):
                rep_output_dir = os.path.join(output_dir, f"R_{replica:04d}")
                if not os.path.exists(rep_output_dir):
                    os.makedirs(rep_output_dir)

        # Store directory paths and simulation metadata
        self.output_dir = output_dir
        self.root_output_dir = output_dir
        self.runname = runname
        self.n_replicas = n_replicas

        # If True, always accept reactive paths (used for reweighting schemes)
        self.always_accept = always_accept

        # Initialize simulation state and output containers
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

        # Flag indicating whether equilibration has finished
        self._equilibrated = False

        # Initialize output variables
        self.trajectory_ensemble = []  # Stores sampled trajectories
        self.force_evaluations = []  # Stores number of force evaluations
        self.lengths = []  # Stores trajectory lengths

        # Additional statistics for always-accept sampling
        if self.always_accept:
            self.accepted_weights_lengths = []
            self.accepted_p_sel_lengths = []
            self.accepted_p_sel_norm_lengths = []
            self.accepted_weights_trajectory_ensemble = []
            self.accepted_p_sel_trajectory_ensemble = []
            self.accepted_p_sel_norm_trajectory_ensemble = []
            self.total_weights = []
            self.total_p_sel = []
            self.total_p_sel_norm = []
            self.accepted_shooting_point_indexes = []

        # Shooting point statistics
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

        # Compute shooting point selection probabilities
        p_sel_current_path = self.p_sel.compute(self.current_path)
        p_sel_current_path_norm = p_sel_current_path.sum()

        # Normalize selection weights
        w = p_sel_current_path/p_sel_current_path_norm
        
        tqdm_title = "" if replica < 0 else f"Replica {replica+1:04d}"

        # Main TPS loop
        for trial in tqdm(range(n_trials), desc = tqdm_title, unit="trials", disable=(not show_progress)):

            # Select shooting point from current path
            shooting_point_index = np.random.choice(np.arange(0, self.current_path.shape[0]), p=w)
            shooting_point = self.current_path[shooting_point_index]

            if self._equilibrated:
                self.total_shooting_points.append(shooting_point)

            # Randomly choose shooting direction (forward or backward)
            shooting_direction = np.random.choice([1,0])

            # Determine which state the endpoints belong to
            rv_in_A = self.state_A.is_in_state(self.current_path[0])
            rv_in_B = self.state_B.is_in_state(self.current_path[0])
            fw_in_A = self.state_A.is_in_state(self.current_path[-1])
            fw_in_B = self.state_B.is_in_state(self.current_path[-1])

            # Generate forward or backward trajectory depending on shooting direction
            if shooting_direction == 1:

                reverse_trj = self.current_path[:shooting_point_index]
                force_eval_rv = 0
                
                forward_trj, force_eval_fw = self.generate_path(
                    shooting_point=shooting_point, 
                    max_path_length=self.max_path_length - (reverse_trj.shape[0] + 1), 
                    configuration_output_frequency=configuration_output_frequency)

                fw_in_A = self.state_A.is_in_state(forward_trj[-1])
                fw_in_B = self.state_B.is_in_state(forward_trj[-1])

            else:
        
                forward_trj = self.current_path[shooting_point_index:]
                force_eval_fw = 0
                
                reverse_trj, force_eval_rv = self.generate_path(
                    shooting_point=shooting_point, 
                    max_path_length=self.max_path_length - forward_trj.shape[0], 
                    configuration_output_frequency=configuration_output_frequency)
    
                rv_in_A = self.state_A.is_in_state(reverse_trj[-1])
                rv_in_B = self.state_B.is_in_state(reverse_trj[-1])

            # Check if endpoints lie in opposing states
            is_reactive_path = (fw_in_A and rv_in_B) or (fw_in_B and rv_in_A)

            # If path is reactive evaluate acceptance
            if is_reactive_path:

                if self._equilibrated: 
                    self.reactive_trials +=1

                # Construct proposed trajectory
                if shooting_direction == 1:
                    proposed_path = np.vstack([reverse_trj, forward_trj])
                else:
                    proposed_path = np.vstack([reverse_trj[::-1], forward_trj[1:]])

                # Compute selection probabilities for proposed path
                p_sel_proposed_path = self.p_sel.compute(proposed_path)
                p_sel_proposed_path_norm = p_sel_proposed_path.sum()
                
                # Accept according to selected scheme
                if self.always_accept:
                    self.current_path = proposed_path
                    p_sel_current_path = p_sel_proposed_path
                    p_sel_current_path_norm = p_sel_proposed_path_norm
                    w = p_sel_current_path/p_sel_current_path_norm

                    if self._equilibrated:
                        if shooting_direction == 1:
                            self.accepted_shooting_points_fw.append(shooting_point)
                            new_shooting_point_index = len(reverse_trj)
                        else:
                            self.accepted_shooting_points_bw.append(shooting_point)
                            new_shooting_point_index = len(reverse_trj)-1

                elif np.random.random() < p_sel_current_path_norm / p_sel_proposed_path_norm:
                    self.current_path = proposed_path
                    p_sel_current_path = p_sel_proposed_path
                    p_sel_current_path_norm = p_sel_proposed_path_norm
                    w = p_sel_current_path/p_sel_current_path_norm

                    if self._equilibrated:
                        self.accepted_trials += 1
                        if shooting_direction == 1:
                            self.accepted_shooting_points_fw.append(shooting_point)
                        else:
                            self.accepted_shooting_points_bw.append(shooting_point)

                else:
                    if self._equilibrated:
                        self.rejected_d2lr_trials += 1

            else:
                if self.always_accept:
                    new_shooting_point_index = shooting_point_index
                
            # Store trajectory information after equilibration
            if self._equilibrated:

                if path_length_output_frequency > 0:
                    if trial % path_length_output_frequency == 0:
                        self.lengths.append(self.current_path.shape[0])

                        if self.always_accept:
                            self.accepted_shooting_point_indexes.append(shooting_point_index)
                            self.accepted_weights_lengths.append(w[new_shooting_point_index])
                            self.accepted_p_sel_lengths.append(p_sel_current_path[new_shooting_point_index])
                            self.accepted_p_sel_norm_lengths.append(p_sel_current_path_norm)

                if force_evaluation_output_frequency > 0:
                    if trial % force_evaluation_output_frequency == 0:
                        self.force_evaluations.append(force_eval_fw + force_eval_rv)

                if path_output_frequency > 0:
                    if trial % path_output_frequency == 0:
                        self.trajectory_ensemble.append(self.current_path)

                        if self.always_accept:
                            self.accepted_weights_trajectory_ensemble.append(w[new_shooting_point_index])
                            self.accepted_p_sel_trajectory_ensemble.append(p_sel_current_path[new_shooting_point_index])
                            self.accepted_p_sel_norm_trajectory_ensemble.append(p_sel_current_path_norm)

                # Periodically dump outputs
                if dump_output_frequency > 0:
                    if trial % dump_output_frequency == 0:
                        self.dump_output(trial)

            # Mark equilibration completion
            if trial == equilibration_trials:
                self._equilibrated = True
                self.equilibration_trials += equilibration_trials

        # Final dump
        self.dump_output(n_trials, paths=dump_paths, histo=dump_histo, screen=(replica < 0), free_memory=False)

        self.total_trials += n_trials

        if replica >=0:
            self.init_path = self.current_path

        return
    

    # Function writing simulation output to disk
    def dump_output(self, trial, paths=True, histo=True, screen=False, free_memory=False):
            
            total_sampling_trials = self.total_trials + trial - self.equilibration_trials

            # Save shooting point statistics
            np.savetxt(os.path.join(self.output_dir, f"tot_sp_{self.runname}.txt"), np.array(self.total_shooting_points))
            np.savetxt(os.path.join(self.output_dir, f"acc_sp_fw_{self.runname}.txt"), np.array(self.accepted_shooting_points_fw))
            np.savetxt(os.path.join(self.output_dir, f"acc_sp_bw_{self.runname}.txt"), np.array(self.accepted_shooting_points_bw))

            # Save additional statistics for always_accept mode
            if self.always_accept:
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

            # Compute and save histogram of sampled configurations
            if histo:
                x_min, x_max = -2.75, 2.75
                y_min, y_max = -2.75, 2.75
                num_bins = 100

                all_positions = np.vstack(self.trajectory_ensemble)

                # Optional symmetry operation
                if self._symmetric:
                    reflected_positions = all_positions.copy()
                    reflected_positions[:, 1] *= -1
                    all_positions = np.vstack((all_positions, reflected_positions))

                Hist, xedges, yedges = np.histogram2d(
                    all_positions[:, 0],
                    all_positions[:, 1],
                    bins=num_bins,
                    range=[[x_min, x_max], [y_min, y_max]],
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

            # Save final trajectory
            np.savetxt(os.path.join(self.output_dir, f"final_path_{self.runname}.txt"), self.current_path)
            np.savetxt(os.path.join(self.root_output_dir, f"final_path_{self.runname}.txt"), self.current_path)
            
            # Optionally free memory
            if free_memory:
                self.trajectory_ensemble.clear()
                del self.trajectory_ensemble
                gc.collect()