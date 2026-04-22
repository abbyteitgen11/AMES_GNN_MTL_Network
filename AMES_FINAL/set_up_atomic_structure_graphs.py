
import re
from typing import Dict, List, Tuple, Union

from atomic_structure_graphs import AtomicStructureGraphs
from XG_graphs import XG

def set_up_atomic_structure_graphs(
    graph_type: str,
    species: List[str],
    bond_angle_feature: bool,
    dihedral_angle_feature: bool,
    spec_features: List[str],
    n_max_neighbours: int = 12,
    distance_features=None,
) -> Tuple[AtomicStructureGraphs]:

    r"""

    AtomicStructure(Hetero)Graphs factory.

      :param graph_type (str): specifies the type of graph to be constructed

          - graph_type = 'geometric': a geometric graph construction in which
                  edges are set up between the nearest n_max_neighbours of
                  every node; n_max_neighbours is really a minimum number of
                  neighbours, because the graph must be undirected, so it may
                  be that additional neighbours are added in order to
                  ensure undirectedness.

          - graph_type = 'covalent': this is the 'chemical' graph rep., in
                  which an edge corresponds to a chemical bond; edges
                  are placed between nodes separated by a distance
                  equal or smaller than the sum of covalent radii
                  times alpha, i.e. rij < alpha(rci + rcj); again the
                  graph is undirected, so every bond is represented as
                  two edges.

          - graph_type = 'generalised': this is different to the previous two, in
                  that the graph contains also a line graph for the
                  bond angles (identified by bond_angle_index and
                  bond_angle_attr), and optionally, a second lineline
                  graph for dihedral angles. This is as yet experimental.

      :param species List[str]: the list of chemical species seen in the database

      :param spec_features List[str]: a list of Mendeleev-recognised keywords identifying
                  chemical species properties (e.g. 'atomic_number', 'covalent_radius', etc).
                  Two special cases of non-Mendeleev keys are accepted, namely 'group' and/or 
                  'period'; if either (or both, but redundant) of these keys is given, then
                  to the list of node features two 1D one-hot encoded vectors will be added, 
                  one of length 7 (with a 1 at the entry corresponding to the element period and
                  zeros elsewhere), and one of length 18 (with a 1 at the entry of the element
                  group). Therefore, using 'group|period' adds 25 features to the nodes (ions). 

      :param features dict: a dictionary containing additional parameters that may be
          needed to be passed to specific subclasses of graph or heterograph. Among these
          some will be:

          - features['edge_features'] (Features): defines edge features between nodes
          - features['bond_angle_features'] (Features): defines bond angle features on nodes
          - features['dihedral_features'] (Features): optional, defines dihedral angle features on edges
          - features['alpha'] (float): defines the cut-off between nodes (ions) of two species 1 and 2
                                   as rc = alpha * (rc1 + rc2) where rc is the cutoff, and rc1,2 are
                                   the covalent radii of species 1 and 2. 
          - features['n_max_neighbours'] (int): self-explanatory; used e.g. in QM9_XG_graphs

      :param pooling str: the type of pooling to perform by the model, can be 'add' or
          'mean'; the latter is appropriate for energy-per-atom regression, 'add' for 
          total energy regression. 'add' is the default.


    """

    graphs = XG(
        species_list=species,
        bond_angle_feature=bond_angle_feature,
        dihedral_angle_feature=dihedral_angle_feature,
        node_feature_list=spec_features,
        n_max_neighbours=n_max_neighbours,
        distance_features=distance_features,
    )


    return graphs
