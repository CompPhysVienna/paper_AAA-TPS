import numpy as np  # Numerical library for arrays and random sampling
from functools import partial  # Allows fixing arguments of a function
from tqdm import tqdm  # Progress bar for loops
import os  # File and directory handling
import pickle  # Serialization for saving trajectory ensembles
import gc  # Garbage collector for freeing memory


# Class implementing ARA-TPS (Always-Reactive Algorithm for TPS)
class ARA_TPS(object):

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
                 runname="ARA-TPS",
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

        # Check whether path starts in A and ends in B
        starts_in_A = self.state_A.is_in_state(init_path[0])
        ends_in_B = self.state_B.is_in_state(init_path[-1])
        path_A2B = starts_in_A and ends_in_B

        # If not A→B check if it is B→A
        if not path_A2B:
            starts_in_B = self.state_B.is_in_state(init_path[0])
            ends_in_A = self.state_A.is_in_state(init_path[-1])
            is_reactive_path = starts_in_B and ends_in_A

            # Ensure path is reactive
            assert is_reactive_path, "The initial path is not reactive"

            # Optionally reverse the path to enforce A→B orientation
            if force_A2B:
                init_path = init_path[::-1]

        # Check that the path enters each state only once
        in_A = np.array([state_A.is_in_state(x) for x in init_path])
        in_B = np.array([state_B.is_in_state(x) for x in init_path])

        count_A = np.count_nonzero(in_A)
        count_B = np.count_nonzero(in_B)

        assert count_A == 1 and count_B == 1, f"The initial path has more than one point in each state. In A: {count_A}. In B: {count_B}"

        # Store the initial trajectory
        self.init_path = init_path

        # Bind state list to the path generation function
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


    # Helper function to change output directory (used for replicas)
    def _set_output_dir(self, output_dir):

        self.output_dir = output_dir

        # Create directory if necessary
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        return
    

    # Initialize simulation variables and storage containers
    def _initialize(self):
        
        # Save the initial path
        np.savetxt(os.path.join(self.output_dir, f"init_path_{self.runname}.txt"), self.init_path)

        # Set current path
        self.current_path = self.init_path.copy()

        # Flag indicating whether equilibration finished
        self._equilibrated = False

        # Output containers
        self.trajectory_ensemble = []  # List storing sampled trajectories
        self.force_evaluations = []  # List storing number of force evaluations
        self.lengths = []  # List storing path lengths

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

        # Configure replica output directory
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

        # Compute selection probabilities
        p_sel_current_path = self.p_sel.compute(self.current_path)
        p_sel_current_path_norm = p_sel_current_path.sum()

        # Normalize weights
        w = p_sel_current_path / p_sel_current_path_norm

        tqdm_title = "" if replica < 0 else f"Replica {replica+1:04d}"

        # Main TPS loop
        for trial in tqdm(range(n_trials), desc=tqdm_title, unit="trials", disable=(not show_progress)):

            # Select shooting point
            shooting_point_index = np.random.choice(np.arange(0, self.current_path.shape[0]), p=w)
            shooting_point = self.current_path[shooting_point_index]
            
            if self._equilibrated:
                self.total_shooting_points.append(shooting_point)

            # Split current path at shooting point
            backward_trj = self.current_path[:shooting_point_index]
            forward_trj = self.current_path[shooting_point_index:]

            # Generate new half trajectory
            half_trj, force_eval = self.generate_path(
                shooting_point=shooting_point, 
                max_path_length=self.max_path_length - max(forward_trj.shape[0], (backward_trj.shape[0] + 1)), 
                configuration_output_frequency=configuration_output_frequency)

            # Determine final state of new trajectory
            ends_in_A = self.state_A.is_in_state(half_trj[-1])
            ends_in_B = self.state_B.is_in_state(half_trj[-1])

            # If trajectory reaches a stable state it is reactive
            if ends_in_A or ends_in_B:

                if self._equilibrated:
                    self.reactive_trials += 1

                # Construct proposed path
                if ends_in_B:
                    proposed_path = np.vstack([backward_trj, half_trj])
                else:
                    proposed_path = np.vstack([half_trj[::-1], forward_trj[1:]])

                # Compute selection probabilities
                p_sel_proposed_path = self.p_sel.compute(proposed_path)
                p_sel_proposed_path_norm = p_sel_proposed_path.sum()

                # Acceptance probability
                p_sel_ratio = p_sel_current_path_norm / p_sel_proposed_path_norm                
                
                # Accept or reject
                if np.random.random() < p_sel_ratio:

                    self.current_path = proposed_path
                    p_sel_current_path = p_sel_proposed_path
                    p_sel_current_path_norm = p_sel_proposed_path_norm
                    w = p_sel_current_path / p_sel_current_path_norm

                    if self._equilibrated:
                        self.accepted_trials += 1
                        if ends_in_B:
                            self.accepted_shooting_points_fw.append(shooting_point)
                        else:
                            self.accepted_shooting_points_bw.append(shooting_point)

                else:
                    if self._equilibrated:
                        self.rejected_d2lr_trials += 1

            # Store path statistics after equilibration
            if self._equilibrated:

                if path_length_output_frequency > 0 and trial % path_length_output_frequency == 0:
                    self.lengths.append(self.current_path.shape[0])

                if force_evaluation_output_frequency > 0 and trial % force_evaluation_output_frequency == 0:
                    self.force_evaluations.append(force_eval)

                if path_output_frequency > 0 and trial % path_output_frequency == 0:
                    self.trajectory_ensemble.append(self.current_path)

                # Periodically dump outputs
                if dump_output_frequency > 0 and trial % dump_output_frequency == 0:
                    self.dump_output(trial)

            # Mark end of equilibration
            if trial == equilibration_trials:
                self._equilibrated = True
                self.equilibration_trials += equilibration_trials

        # Final output dump
        self.dump_output(n_trials, paths=dump_paths, histo=dump_histo, screen=(replica < 0), free_memory=False)

        self.total_trials += n_trials

        if replica >= 0:
            self.init_path = self.current_path

        return
    

    # Function to write simulation outputs to disk
    def dump_output(self, trial, paths=True, histo=True, screen=False, free_memory=False):
            
            total_sampling_trials = self.total_trials + trial - self.equilibration_trials

            # Save shooting point statistics
            np.savetxt(os.path.join(self.output_dir, f"tot_sp_{self.runname}.txt"), np.array(self.total_shooting_points))
            np.savetxt(os.path.join(self.output_dir, f"acc_sp_fw_{self.runname}.txt"), np.array(self.accepted_shooting_points_fw))
            np.savetxt(os.path.join(self.output_dir, f"acc_sp_bw_{self.runname}.txt"), np.array(self.accepted_shooting_points_bw))

            # Save path statistics
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

            # Save final path
            np.savetxt(os.path.join(self.output_dir, f"final_path_{self.runname}.txt"), self.current_path)
            np.savetxt(os.path.join(self.root_output_dir, f"final_path_{self.runname}.txt"), self.current_path)

            # Optionally free memory
            if free_memory:
                self.trajectory_ensemble.clear()
                del self.trajectory_ensemble
                gc.collect()