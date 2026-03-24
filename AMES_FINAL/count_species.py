from collections import Counter

from ISSSTY_utils import read_ISSSTY_structure

def count_species(file_paths):
    """
    Summarizes atom types across multiple files

    Args:
        file_paths (list): List of file paths containing atom data.

    Returns:
        dict: A dictionary with atom types as keys and their counts as values.
    """
    #overall_counts = Counter()  # To store the total counts of atom types
    n_atoms_overall = 0

    for file_path in file_paths:
        # Read the list of atoms from the file
        (   molecule_id,
            n_atoms,
            labels,
            positions,
            toxicity
        ) = read_ISSSTY_structure(file_path)

        #n_atoms_overall.append(n_atoms)
        if n_atoms > n_atoms_overall:
            n_atoms_overall = n_atoms

        # Count the atoms in the current file and update the overall count
        #file_counts = Counter(labels)
        #overall_counts.update(file_counts)

        #atom_counts = Counter(n_atoms)
        #overall_atom_counts.update(atom_counts)

    return n_atoms_overall