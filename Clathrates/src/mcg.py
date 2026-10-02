import numpy as np
import commons as params
from numba import jit
import mdtraj as md
import networkx as nx
import re
from scipy import linalg
import freud
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def COM_calculation_gro(file):
    """Parse a .gro file and calculate the center of mass for CO2 (UNL) and water (SOL)."""
    C_M = params.C_M  # Carbon atomic mass
    O_M = params.O_M  # Oxygen atomic mass
    H_M = params.H_M  # Hydrogen atomic mass

    CAR_positions = []
    WATER_positions = []

    with open(file, 'r') as f:
        lines = f.readlines()
    
    atom_lines = lines[2:-1]
    box_line = lines[-1].strip()
    box_length = float(box_line.split()[0])  # Assuming a cubic box
    
    for line in atom_lines:
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        x = float(line[20:28].strip())
        y = float(line[28:36].strip())
        z = float(line[36:44].strip())

        # Collect CO2 atoms (UNL residue)
        if residue_name == "UNL":
            CAR_positions.append((atom_name, np.array([x, y, z])))
        
        # Collect water atoms (SOL residue)
        elif residue_name == "SOL":
            WATER_positions.append((atom_name, np.array([x, y, z])))
    
    # Calculate COM for CO2 (UNL)
    CAR_COM = []
    for i in range(0, len(CAR_positions), 3):  # Groups of 3 atoms
        co_pos = CAR_positions[i][1]
        o1_pos = CAR_positions[i + 1][1]
        o2_pos = CAR_positions[i + 2][1]
        com = (co_pos * C_M + (o1_pos + o2_pos) * O_M) / (C_M + 2 * O_M)
        CAR_COM.append(com)
    
    # Calculate COM for water (SOL)
    WATER_COM = []
    for i in range(0, len(WATER_positions), 4):  # Groups of 4 atoms
        o_pos = WATER_positions[i][1]
        h1_pos = WATER_positions[i + 1][1]
        h2_pos = WATER_positions[i + 2][1]
        com = (o_pos * O_M + (h1_pos + h2_pos) * H_M) / (O_M + 2 * H_M)
        WATER_COM.append(com)
    
    return np.array(CAR_COM), np.array(WATER_COM), box_length


def CO2ClathrateCOM(positions_nm, box_length):
    """
    Computes centers of mass (COMs) for CO₂ and water molecules directly from OpenMM positions.
    
    Args:
        positions_nm (np.ndarray): Positions from OpenMM (shape: (n_atoms, 3), units: nm).
        box_length (float): Cubic box edge length (nm).
    
    Returns:
        tuple: (CO2_COMs, WATER_COMs, box_length)
              - CO2_COMs: Shape (512, 3), COMs of CO₂ molecules (nm).
              - WATER_COMs: Shape (2944, 3), COMs of water molecules (nm).
              - box_length: Same as input (nm).
    """
    C_M = params.C_M  # Carbon atomic mass
    O_M = params.O_M  # Oxygen atomic mass
    H_M = params.H_M  # Hydrogen atomic mass

    # --- CO₂ Molecules (512 molecules, 3 atoms each: C, O, O) ---
    n_co2 = 512
    co2_positions = positions_nm[:3 * n_co2]  # First 1536 atoms (512 * 3)
    co2_positions = co2_positions.reshape(n_co2, 3, 3)  # Shape: (512, 3, 3)
    
    
    # Compute CO₂ COMs: (C * C_M + O1 * O_M + O2 * O_M) / (C_M + 2 * O_M)
    co2_coms = (co2_positions[:, 0] * C_M + (co2_positions[:, 1] + co2_positions[:, 2]) * O_M) / (C_M + 2 * O_M)

    co2_coms_wrapped = co2_coms - np.floor(co2_coms / box_length) * box_length

    
    # --- Water Molecules (2944 molecules, 4 atoms each: O, H, H, [MW]) ---
    n_waters = 2944
    water_start_idx = 3 * n_co2  # Start after CO₂ atoms
    water_positions = positions_nm[water_start_idx : water_start_idx + 4 * n_waters]  # Shape: (11776, 3)
    water_positions = water_positions.reshape(n_waters, 4, 3)  # Shape: (2944, 4, 3)
    
    # Compute water COMs: (O * O_M + H1 * H_M + H2 * H_M) / (O_M + 2 * H_M)
    # Ignore the 4th atom (MW virtual site) if present
    water_coms = (water_positions[:, 0] * O_M + (water_positions[:, 1] + water_positions[:, 2]) * H_M) / (O_M + 2 * H_M)

    water_coms_wrapped = water_coms - np.floor(water_coms / box_length) * box_length

    return co2_coms_wrapped, water_coms_wrapped, box_length

def COM_calculation_gro_multiframe(file):

    """Parse a multi-frame .gro file and calculate the COM and MCG_OP for each frame."""
    C_M = params.C_M  # Carbon atomic mass
    O_M = params.O_M  # Oxygen atomic mass
    H_M = params.H_M  # Hydrogen atomic mass

    # Data structures to store results
    all_car_com = []
    all_water_com = []
    all_box_lengths = []
    all_steps = []  # Store the 'step' value from the top line of each frame

    with open(file, 'r') as f:
        lines = f.readlines()

    # Identify the structure of the file
    num_atoms = int(lines[1].strip())  # Number of atoms (constant across frames)
    frame_size = num_atoms + 3        # 2 header lines + atom lines + 1 box line
    num_frames = len(lines) // frame_size

    for frame_idx in range(num_frames):
        start_line = frame_idx * frame_size
        end_line = start_line + frame_size
        frame_lines = lines[start_line:end_line]

        # Extract 'step' number from the first line of the frame
        first_line = frame_lines[0].strip()
        step_match = re.search(r'step=\s*(\d+)', first_line)  # Regex to extract number after 'step='
        if step_match:
            step_value = int(step_match.group(1))  # Extracted step value
            all_steps.append(step_value)
        else:
            print(f"Warning: Could not find step value in frame {frame_idx}")

        CAR_positions = []
        WATER_positions = []

        # Parse each frame
        for line in frame_lines[2:-1]:  # Skip header and box line
            residue_name = line[5:10].strip()
            atom_name = line[10:15].strip()
            x = float(line[20:28].strip())
            y = float(line[28:36].strip())
            z = float(line[36:44].strip())

            if residue_name == "UNL":
                CAR_positions.append((atom_name, np.array([x, y, z])))
            elif residue_name == "SOL":
                WATER_positions.append((atom_name, np.array([x, y, z])))

        # Calculate COM for CAR (CO2)
        CAR_COM = []
        for i in range(0, len(CAR_positions), 3):
            co_pos = CAR_positions[i][1]
            o1_pos = CAR_positions[i + 1][1]
            o2_pos = CAR_positions[i + 2][1]
            com = (co_pos * C_M + (o1_pos + o2_pos) * O_M) / (C_M + 2 * O_M)
            CAR_COM.append(com)

        # Calculate COM for WATER (SOL)
        WATER_COM = []
        for i in range(0, len(WATER_positions), 4):
            o_pos = WATER_positions[i][1]
            h1_pos = WATER_positions[i + 1][1]
            h2_pos = WATER_positions[i + 2][1]
            com = (o_pos * O_M + (h1_pos + h2_pos) * H_M) / (O_M + 2 * H_M)
            WATER_COM.append(com)

        # Get box length from the last line of the frame
        box_line = frame_lines[-1].strip()
        box_length = float(box_line.split()[0])  # Assuming a cubic box

        # Store the data for the current frame
        all_car_com.append(np.array(CAR_COM))
        all_water_com.append(np.array(WATER_COM))
        all_box_lengths.append(box_length)

    return np.array(all_steps), all_car_com, all_water_com, all_box_lengths

def COM_calculation_mdtraj(traj):
    """Calculate the center of mass (COM) for CO2 and water molecules from an mdtraj trajectory."""
    C_M = params.C_M  # Carbon atomic mass
    O_M = params.O_M  # Oxygen atomic mass
    H_M = params.H_M  # Hydrogen atomic mass
    
    # Data structures to store results
    all_car_com = []
    all_water_com = []
    all_box_lengths = []
    # all_steps = np.arange(traj.n_frames)  # Use frame index as step
    timesps = traj.time
    
    # Identify atom indices
    topology = traj.topology
    CAR_indices = []  # CO2 indices
    WATER_indices = []  # Water indices
    
    for residue in topology.residues:
        if residue.name == "UNL" or residue.name == "CAR":  # CO2 molecule
            atoms = [atom.index for atom in residue.atoms]
            if len(atoms) == 3:  # Ensure CO2 has exactly 3 atoms
                CAR_indices.append(atoms)
        elif residue.name == "HOH" or residue.name == "water":  # TIP4P/Ice water molecule
            oxygen = None
            hydrogens = []
            for atom in residue.atoms:
                if atom.element.symbol == "O":
                    oxygen = atom.index
                elif atom.element.symbol == "H":
                    hydrogens.append(atom.index)
            if oxygen is not None and len(hydrogens) == 2:  # Ignore MW/EP site
                WATER_indices.append([oxygen] + hydrogens)
    
    for frame in range(traj.n_frames):
        CAR_COM = []
        WATER_COM = []
        
        xyz = traj.xyz[frame]  # Positions of all atoms in this frame
        
        # Compute CO2 COM
        for indices in CAR_indices:
            co_pos = xyz[indices[0]]  # Carbon
            o1_pos = xyz[indices[1]]  # Oxygen 1
            o2_pos = xyz[indices[2]]  # Oxygen 2
            com = (co_pos * C_M + (o1_pos + o2_pos) * O_M) / (C_M + 2 * O_M)
            CAR_COM.append(com)
        
        # Compute Water COM
        for indices in WATER_indices:
            o_pos = xyz[indices[0]]  # Oxygen
            h1_pos = xyz[indices[1]]  # Hydrogen 1
            h2_pos = xyz[indices[2]]  # Hydrogen 2
            com = (o_pos * O_M + (h1_pos + h2_pos) * H_M) / (O_M + 2 * H_M)
            WATER_COM.append(com)
        
        # Extract box length (assuming cubic box)
        box_length = traj.unitcell_lengths[frame, 0]
        
        all_car_com.append(np.array(CAR_COM))
        all_water_com.append(np.array(WATER_COM))
        all_box_lengths.append(box_length)
    
    return timesps, all_car_com, all_water_com, all_box_lengths


def is_within_cone_torch(guest1_pos, guest2_pos, water_pos, box_length, angle=np.pi/2):
    """
    Vectorized cone check using PyTorch (GPU-compatible)
    """
    # Convert to tensors if not already
    guest1_pos = torch.as_tensor(guest1_pos)
    guest2_pos = torch.as_tensor(guest2_pos)
    water_pos = torch.as_tensor(water_pos)
    
    # Calculate vectors with PBC
    g1_to_g2 = guest2_pos - guest1_pos
    g1_to_water = water_pos - guest1_pos
    g2_to_water = guest2_pos - water_pos
    
    # Apply periodic boundary conditions
    g1_to_g2 -= torch.round(g1_to_g2 / box_length) * box_length
    g1_to_water -= torch.round(g1_to_water / box_length) * box_length
    g2_to_water -= torch.round(g2_to_water / box_length) * box_length
    
    # Normalize
    g1_to_g2 = g1_to_g2 / torch.norm(g1_to_g2, dim=-1, keepdim=True)
    g1_to_water = g1_to_water / torch.norm(g1_to_water, dim=-1, keepdim=True)
    g2_to_water = g2_to_water / torch.norm(g2_to_water, dim=-1, keepdim=True)
    
    # Compute angles
    angle_g1 = torch.acos(torch.clamp(torch.sum(g1_to_g2 * g1_to_water, dim=-1), -1.0, 1.0))
    angle_g2 = torch.acos(torch.clamp(torch.sum(g1_to_g2 * g2_to_water, dim=-1), -1.0, 1.0))


    return (angle_g1 <= (angle / 2)) & (angle_g2 <= (angle / 2))

def largest_cluster_size_opt(adj_matrix):
    """Optimized cluster finding using scipy.sparse.csgraph"""
    
    n_components, labels = connected_components(csr_matrix(adj_matrix))
    if n_components == 0:
        return 0, []
    
    cluster_sizes = np.bincount(labels)
    largest_label = np.argmax(cluster_sizes)
    return cluster_sizes[largest_label], np.where(labels == largest_label)[0]

def MCG_optimized(guest_pos, water_pos, box_length, order=1, guest_cutoff=0.9, water_cutoff=0.6):
    """
    Optimized MCG calculation with:
    - Vectorized cone checks
    - Batched operations
    - Reduced memory allocations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # print(f"Running on {device}")
    
    # Convert to tensors
    guest_pos = torch.as_tensor(guest_pos, device=device)
    water_pos = torch.as_tensor(water_pos, device=device)
    
    # Neighbor lists with freud (still CPU)
    box = freud.box.Box.cube(box_length)
    guest_nl = freud.locality.AABBQuery(box, guest_pos.cpu().numpy()).query(
        guest_pos.cpu().numpy(), {'r_max': guest_cutoff}
    ).toNeighborList()
    
    water_nl = freud.locality.AABBQuery(box, guest_pos.cpu().numpy()).query(
        water_pos.cpu().numpy(), {'r_max': water_cutoff}
    ).toNeighborList()
    
    # Pre-filter mutual waters
    water_guest_map = {}
    for w, g in zip(water_nl.query_point_indices, water_nl.point_indices):
        water_guest_map.setdefault(g, set()).add(w)
    
    # Process pairs in batches
    adjacency = np.zeros((len(guest_pos), len(guest_pos)), dtype=bool)
    successful_pairs = []
    
    for i, j in zip(guest_nl.point_indices, guest_nl.query_point_indices):
        if i >= j:
            continue
            
        mutual_waters = water_guest_map.get(i, set()) & water_guest_map.get(j, set())
        if len(mutual_waters) < 5:
            continue
            
        # Vectorized cone check
        water_indices = torch.tensor(list(mutual_waters), dtype=torch.long, device=device)
        waters = water_pos[water_indices]
        in_cone = is_within_cone_torch(
            guest_pos[i], guest_pos[j], waters, box_length
        )
        
        if torch.sum(in_cone) >= 5:
            adjacency[i,j] = True
            successful_pairs.append([i, j] + waters[in_cone][:5].cpu().numpy().tolist())
    
    # Find largest cluster
    cluster_size, cluster_indices = largest_cluster_size_opt(adjacency)
    return cluster_size, successful_pairs, guest_pos[cluster_indices].cpu().numpy()
