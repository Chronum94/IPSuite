import numpy as np
from scipy.stats.qmc import Halton
from scipy.spatial.distance import pdist
from ase import Atoms


def _generate_qmc_points(cell_size, num_points, min_bond_length):
    sampler = Halton(3)
    random_points = sampler.random(num_points)
    random_points_physical = random_points * cell_size

    distances = pdist(random_points_physical)

    points_too_close = []
    for i in range(num_points - 1):
        for j in range(i + 1, num_points):
            if distances[num_points * i + j - ((i + 2) * (i + 1)) // 2] < min_bond_length:
                points_too_close.append(i)

    random_points_physical = np.delete(random_points_physical, points_too_close, axis=0)
    return random_points_physical

def generate_qmc_bulk_configs(
    cell_size, 
    species, 
    min_bond_length,
    probabilities=None,
    bond_volume_scale=2.0, 
    n_point_clouds=4,
    n_realizations_per_cloud=4,
):
    probabilities = np.ones(len(species)) / len(species) if probabilities is None else np.array(probabilities) / np.sum(probabilities)

    num_points = int(cell_size**3 / (bond_volume_scale * min_bond_length**3))
    
    atoms_list = []
    for _ in range(n_point_clouds):
        random_points_physical = _generate_qmc_points(cell_size, num_points, min_bond_length)
        for _ in range(n_realizations_per_cloud):
            species_array = np.random.choice(
                species, size=len(random_points_physical), p=probabilities
            )
            d_x = np.ptp(random_points_physical[:, 0])
            d_y = np.ptp(random_points_physical[:, 1])
            d_z = np.ptp(random_points_physical[:, 2])
            x_pad = min_bond_length - (cell_size - d_x)
            y_pad = min_bond_length - (cell_size - d_y)
            z_pad = min_bond_length - (cell_size - d_z)
            cell = np.array([cell_size] * 3) + np.array([x_pad, y_pad, z_pad])
            atoms = Atoms(species_array, random_points_physical, cell=cell, pbc=True)
            atoms.center()
            atoms_list.append(atoms)
    return atoms_list

print(generate_qmc_bulk_configs(13, ['C', 'B', 'N']))