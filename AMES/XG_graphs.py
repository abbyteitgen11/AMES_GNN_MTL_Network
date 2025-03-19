from typing import Dict, List

from mendeleev import element
import numpy as np
from scipy.constants import physical_constants
import torch
from torch_geometric.data import Data

from atomic_structure_graphs import AtomicStructureGraphs
from derivatives import Derivatives
from exceptions import IndexOutOfBounds
from distance_feature import DistanceFeature
from features import Features
from ISSSTY_utils import read_ISSSTY_structure
from structure_utils import get_dihedral_angle


class MyData(Data):

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'neighbour_id':
            return self.x.size(0)
        return super().__inc__(key, value, *args, **kwargs)


class XG(AtomicStructureGraphs):
    """

    A class to read molecule information from the database file
    and convert it to torch_geometric Data graph. In this class, graphs
    are constructed in a chemically intuitive way: a node (atom) has edges
    only to other nodes that are up to a maximum distance away.

    There are several key differences with the scheme of Xie-Grossmann and my
    own earlier implementation:

    1) There are no "geometrical" features of nodes; all the geometry is now
    contained in edge features; this is essential if we want to have a workable
    calculation of derivatives (forces).

    2) Although graphs are undirected (if i has an edge to j, j has an edge to i),
    this does not require that the i->j edge features be the same as those of j->i;
    features that depend on distance will be obviously the same, but edge i->j can
    have features depending on bond-angles with i as central atom, while j->i can
    have features depending on bond-angles with j as the central atom.

    3) The idea is to have only two-three edge features, the first feature depending
    on distance i-j, the second (potentially different for i->j and j->i) is a
    sum of cosines of bond angles, and the third is (optional) a sum of cosines
    of dihedral angles k-i-j-l.

    4) Input edge features, like input node features, will be expanded to any
    desired dimention by a linear layer-plus-activation before entering the main
    convolution loop in the graph NN, so the fact that we are using a relatively small
    number of input features need not be a limitation.

    5) We dump the Xie-Grossman edge-features in favour of a decaying function of
    distance; it will be something similar to a Gaussian with a rate of decay that
    depends on the covalent radii of the two species involved, and it will be
    cut-off smoothly to zero at some maximum cutoff.

    """

    def __init__(
            self,
            species_list: List[str],
            distance_features: Dict,
            bond_angle_feature: bool = False,
            dihedral_angle_feature: bool = False,
            node_feature_list: List[str] = [],
            n_max_neighbours: int = 12,
            alpha: float = 1.1
    ) -> None:

        # initialise the base class
        super().__init__(
            species_list=species_list,
            distance_features=distance_features,
            bond_angle_feature=bond_angle_feature,
            dihedral_angle_feature=dihedral_angle_feature,
            node_feature_list=node_feature_list,
        )

        self.n_max_neighbours = n_max_neighbours
        self.alpha = alpha  # alpha is the scaling factor for bond (edge)
        # critera, i.e. two atoms are bonded if their
        # separation is r <= alpha*(rc1 + rc2), where
        # rci are the respective covalent radii

        self.covalent_radii = self.get_covalent_radii()

    def get_covalent_radii(self) -> Dict[str, float]:

        """

        Sets up and returns a dictionary of covalent radii (in Ang)
        for the list of species in its argument

        :return: covalent_radii: dict of covalent radius for eash species (in Angstrom)
        :rtype: dict

        """
        covalent_radii = {}

        for label in self.species:
            spec = element(label)
            covalent_radii[label] = spec.covalent_radius / 100.0
            # mendeleev stores radii in pm, hence the factor

        return covalent_radii

    def structure2graph(self, fileName: str, set: tuple) -> Data:

        """

        A function to turn atomic structure information imported from
        a  database file and convert it to torch_geometric Data graph.
        In this particular class graphs are constructed in the following way:

        Edges will be set-up between atoms that are separated by a distance
        equal or smaller than distance_feature.cutoff() (see distance_feature.py).
        Optionally there will be bond-angle features (sum of bond-angle cosines)
        and dihedral-angle features (sum of dihedral angle cosines).

        Args:

        :param: fileName (string): the path to the file where the structure
           information is stored in file.
        :type: str
        :return: graph representation of the structure contained in fileName
        :rtype: torch_geometric.data.Data

        """

        (
            molecule_id,
            n_atoms,
            labels,
            positions,
        ) = read_ISSSTY_structure(fileName)

        # the total number of node features is given by the species features

        n_features = self.spec_features[labels[0]].size

        node_features = torch.zeros((n_atoms, n_features), dtype=torch.float32)

        # atoms will be graph nodes; edges will be created for every
        # neighbour of i within the cutoff distance

        # first we loop over all pairs of atoms and calculate the matrix
        # of squared distances

        dij2 = np.zeros((n_atoms, n_atoms), dtype=float)

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):
                rij = positions[j, :] - positions[i, :]
                rij2 = np.dot(rij, rij)

                dij2[i, j] = rij2
                dij2[j, i] = rij2

        n_neighbours = np.zeros((n_atoms), dtype=int)
        neighbour_distance = np.zeros((n_atoms, self.n_max_neighbours),
                                      dtype=float)
        neighbour_id = np.zeros((n_atoms, self.n_max_neighbours),
                                   dtype=int)

        node0 = []
        node1 = []

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):

                dcut = self.alpha * (
                        self.covalent_radii[labels[i]] + self.covalent_radii[labels[j]]
                )

                dcut2 = dcut * dcut

                if dij2[i, j] <= dcut2:
                    node0.append(i)
                    node1.append(j)

                    node0.append(j)
                    node1.append(i)

                    dij = np.sqrt(dij2[i, j])

                    neighbour_distance[i, n_neighbours[i]] = dij
                    neighbour_distance[j, n_neighbours[j]] = dij

                    neighbour_id[i, n_neighbours[i]] = j
                    neighbour_id[j, n_neighbours[j]] = i

                    n_neighbours[i] += 1
                    n_neighbours[j] += 1

                    if n_neighbours[i] == self.n_max_neighbours or \
                            n_neighbours[j] == self.n_max_neighbours:
                        raise IndexOutOfBounds("n_max_neighbours {} too \
                          small!!!".format(self.n_max_neighbours))

        edge_index = torch.tensor([node0, node1], dtype=torch.long)

        _, num_edges = edge_index.shape

        # get the node features for each atom (node)
        for i in range(n_atoms):
            node_features[i, :] = \
                torch.from_numpy(self.spec_features[labels[i]])

        # now, based on the edge-index information, we can construct the edge attributes

        n_bond_features = 0  # for distance

        if self.bond_angle_feature: n_bond_features += 1  # for bond_angles
        if self.dihedral_angle_feature: n_bond_features += 1  # for dihedral

        bond_features = np.zeros((num_edges, n_bond_features), dtype=float)

        bond_feature_derivatives = np.zeros((1), dtype=float)

        for n in range(num_edges):
            i = edge_index[0, n]
            j = edge_index[1, n]

            dij = np.sqrt(dij2[i, j])
            rij = positions[j, :] - positions[i, :]

            if self.bond_angle_feature:  # include bond angle features
                for nk in range(n_neighbours[i]):
                    k = neighbour_id[i, nk]
                    if k == j: continue  # k must be different from j
                    rik = positions[k, :] - positions[i, :]
                    dik = neighbour_distance[i, nk]

                    cosijk = np.dot(rij, rik) / (dij * dik)
                    # bond_features[n,1] += fij * fik * cosijk
                    bond_features[n, 0] += cosijk #u_k(cosijk)

        if self.dihedral_angle_feature:  # include dihedral features
            for nk in range(n_neighbours[i]):
                k = neighbour_id[i, nk]
                if k == j: continue  # k must be different from j
                rki = positions[i, :] - positions[k, :]
                for nl in range(n_neighbours[j]):
                    l = neighbour_id[j, nl]
                    if l in (k, i): continue  # cannot define dihedral if l==k or l==i
                    rjl = positions[l, :] - positions[j, :]

                    # to define a dihedral angle, atoms k-i-j and i-j-l
                    # must be non-co-linear; if either triad is colinear, then
                    # get_dihedral_angle returns a value of None to avoid ill-defined
                    # dihedral angles

                    coskijl = get_dihedral_angle(rki, rij, rjl)
                    if coskijl: bond_features[n, 1] += coskijl

        spec_id = torch.zeros((n_atoms), dtype=int)

        for n in range(n_atoms):
            spec_id[n] = self.species_dict[labels[n]]

        # now we can create a graph object (Data)
        index = molecule_id
        index_list = set[0].tolist()

        pos = index_list.index(index)  # Get position in the list
        toxicity = [arr[pos] for arr in set[1]]  # Extract from each array

        edge_attr = torch.tensor(bond_features, dtype=torch.float32)
        edge_der = torch.tensor(bond_feature_derivatives, dtype=torch.float32)
        y = torch.tensor(toxicity, dtype=torch.float32).unsqueeze(0)

        pos = torch.from_numpy(positions)

        n_neigh = torch.tensor(n_neighbours, dtype=torch.long)
        neigh_id = torch.tensor(neighbour_id, dtype=torch.long)
        neigh_distance = torch.tensor(neighbour_distance, dtype=torch.float32)

        structure_graph = MyData(
            x=node_features,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            n_neighbours=n_neigh,
            neighbour_id=neigh_id,
            neighbour_distance=neigh_distance,
            spec_id=spec_id
        )

        return structure_graph


# register this derived class as subclass of AtomicStructureGraphs
AtomicStructureGraphs.register(XG)


# below create a class to calculate the derivatives of this model
class XGDerivatives(Derivatives):

    def __init__(
            self,
            species_list,
            distance_features,
            bond_angle,
            dihedral_angle,
    ) -> None:
        super().__init__(
            distance_features=distance_features,
            bond_angle=bond_angle,
            dihedral_angle=dihedral_angle,
        )

        self.species_list = species_list

    def activate_derivatives(self, sample: Data) -> None:
        """
        In order to calculate forces we need the derivatives with
        respect to edge attributes
        """

        sample.edge_attr.requires_grad_(requires_grad=True)