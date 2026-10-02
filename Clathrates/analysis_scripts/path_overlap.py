"""Path-overlap analysis for transition path sampling.

Given the sequence of shooting moves along a TPS run, this module
computes how much of a reference transition path survives as the Markov
chain evolves. The surviving fraction is what the paper calls the
*fractional overlap* between a reference path and subsequent
trajectories (Fig. 2C and Fig. 3B).

The overlap is measured at the level of path segments, not individual
particles: it tracks which frames of the reference path are still
present in the current trajectory after each shooting move. It is not a
coordinate- or atom-based comparison.

Standalone utility. There is no entry point and no CLI. Import
``find_age_of_oldest_path`` and call it with your own data.

Notes on the return values
--------------------------

``find_age_of_oldest_path`` returns two lists.

``LOPS_list``
    One sublist per starting accepted path. Each sublist has one entry
    per *subsequent trial* (accepted or rejected), giving the fraction
    of the starting path that is still present in the current
    trajectory at that trial. Values start at 1.0, shrink when a trial
    regenerates part of the segment, and are frozen on rejected trials.
    The sublist terminates when the surviving material reaches zero.

``ages_list``
    One entry per starting path that dies within the run: the number of
    trials until its material is completely gone. Note that this counts
    *all* trials, not just accepted ones — the counter is incremented on
    both accepted and rejected moves.

Caveats
-------

* The x-axis implied by these lists is the trial index, not the number
  of force evaluations. The paper plots overlap against force
  evaluations; to reproduce that, the caller must accumulate the cost
  of each trial separately and use those values as x-coordinates.

* The name ``find_age_of_oldest_path`` predates the current
  terminology. The function is retained under its original name for
  continuity, but the quantity it returns is the fractional overlap
  curve, not an age.
"""


def find_age_of_oldest_path(indices, directions, lengths, force_eval, trial_results):
    '''
        Parameters:
            indices : list of tuples
                (old_index, new_index) are shooting point indices on the old and new path
            directions : list of int
                (+1 = forward shooting, -1 = backward shooting)
            first_path_length : int
                Length of the initial transition path (number of frames)
            trial_results : list
                boolean list of trial outcome (Accepted = True, Rejected = False)
        Returns:
            LOPS_list : list of lists
                Evolution of the oldest path segment lengths over successive accepted paths.
                Each sublist corresponds to one starting position in AOPS_list.
            AOPS_list : list of int
                Ages (in number of accepted paths) of the oldest path segment for each starting trajectory
    '''

    LOPS_list = []  #lengths of oldest path segments
    ages_list = []

    # loop over all possible paths as starting paths
    for initial_path_idx in range(len(trial_results)):  
        print(f'starting from initial path {initial_path_idx}')
        if trial_results[initial_path_idx]  == True:     #only start from accepted paths
            print('Successful')
            L0 = lengths[initial_path_idx]
    
            SOPS = 0
            EOPS = L0
    
            LOPS = EOPS - SOPS
            Ls_current_initial_path = [LOPS / L0]
            age_current_initial_path = 0
        
            for current_path_idx, trial_result in enumerate(trial_results[initial_path_idx + 1:], start = initial_path_idx + 1):
                print(f'\nComparing to current path {current_path_idx}')
                if trial_result == True:  #path accepted

                    print('Comp path successful')
                
                    (o, n) = indices[current_path_idx]
                    direction = directions[current_path_idx]
                    
                    if direction == 1:        #no shift of indices necessary
                        if n <= EOPS:
                            EOPS = n
                        #otherwise the end stays where it was before
                
                    elif direction == -1:     #indices need to be shifted
                        shift = n - o
                        if o >= SOPS:
                            SOPS = n
                            EOPS += shift
                        else:                        # length stays the same but indices need to be shifted to fit current path 
                            SOPS += shift
                            EOPS += shift

                    print(f'Comp path SOPS {SOPS} EOPS {EOPS}')
        
                    if EOPS - SOPS <= 0:
                        ages_list.append(age_current_initial_path)
                        LOPS_list.append(Ls_current_initial_path)
                        print('Path dies')
                        break
                        
                    else:
                        LOPS = EOPS - SOPS
                        print(f'alive, adding LOPS {LOPS / L0}')
                        Ls_current_initial_path.append(LOPS / L0)
                        age_current_initial_path += 1

                else: # rejected
                    print(f'Rejected, appending LOPS {LOPS / L0}')
                    Ls_current_initial_path.append(LOPS / L0)
                    age_current_initial_path += 1
                
            else:  #hit the limit of max paths
                LOPS_list.append(Ls_current_initial_path)

    return LOPS_list, ages_list