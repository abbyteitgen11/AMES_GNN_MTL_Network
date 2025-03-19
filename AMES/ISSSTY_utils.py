import re
from typing import List, Tuple

import numpy as np
import csv
from load_data import load_data

def read_ISSSTY_structure(
       file_name: str 
    ) -> Tuple[int, int, List[str], float, float]:


    """

    This function opens a file of the ISSSTY database and processes it,
    returning the molecule structure in xyz format and a molecule identifier
    (tag)
    Also opens corresponding csv file to get toxicity data

    Args:

    :param str file_name: filename containing the molecular information

    :return: molecule_id (int): integer identifying the molecule number
        in the database n_atoms (int): number of atoms in the molecule
        species (List[str]): the species of each atom (len = n_atoms)
        coordinates (np.array(float)[n_atoms,3]): atomic positions
        toxicity (int): toxicity of the molecule (-1, 0, or 1)
    :rtype: Tuple[int, int, List[str], float, int]

    """

    with open(file_name, "r") as file_in:
        lines = file_in.readlines()

    n_atoms = int(lines[0])  # number of atoms is specified in 1st line

    words = lines[1].split('_')

    molecule_id = int(words[0])

    #molecular_data = np.array(words[2:], dtype=float)

    species = []  # species label
    coordinates = np.zeros((n_atoms, 3), dtype=float)  # coordinates in Angstrom
    # charge = np.zeros((n_atoms), dtype=float)  # Mulliken charges (e)

    # below extract chemical labels, coordinates and charges

    m = 0

    for n in range(2, n_atoms + 2):

        line = re.sub(
            r"\*\^", "e", lines[n]
        )  # this prevents stupid exponential lines in the data base

        words = line.split()

        species.append(words[0])

        x = float(words[1])
        y = float(words[2])
        z = float(words[3])

        # c = float(words[4])

        coordinates[m, :] = x, y, z

        # charge[m] = c
    
        m += 1

    return molecule_id, n_atoms, species, coordinates