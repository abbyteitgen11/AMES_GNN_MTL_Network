
import re
from typing import Dict, List, Tuple, Union

from atomic_structure_graphs import AtomicStructureGraphs
#from atomic_structure_heterographs import AtomicStructureHeteroGraphs
#from generalised_molecular_graphs import GeneralisedMolecularGraphs
#from geometric_molecular_graphs import GeometricMolecularGraphs
#from pbc_graphs import PBCGraphs
#from QM9_ERH_graphs import QM9ERH
#from QM9_ERH_graphs import QM9ERHDerivatives
#from XG_graphs_global import XG
#from XG_graphs_global import XG
from XG_graphs import XG
#from QM9_XG_graphs import QM9XieGrossmanGraphs
#from QM9_simple_XG_graphs import QM9SimpleXieGrossmanGraphs
#from QM9_simple_XG_graphs import QM9SimpleXieGrossmanDerivatives
#from covalent_molecular_heterographs import CovalentMolecularHeteroGraphs

def set_up_atomic_structure_graphs(
    graph_type: str,
    species: List[str],
    bond_angle_feature: bool,
    dihedral_angle_feature: bool,
    spec_features: List[str],
    n_max_neighbours: int = 12,
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

    #if re.match('^geo', graph_type):

           # graphs = GeometricMolecularGraphs(
           #     species_list = species,
           #     edge_features = features['edge_features'],
           #     bond_angle_features = features['bond_angle_features'],
           #     dihedral_features = features.get('dihedral_features', None),
           #     node_feature_list = spec_features,
           #     n_max_neighbours = features.get('n_max_neighbours', None),
           # )

           # derivatives = None

    #pass

    #elif re.match('^XG', graph_type):

    #graphs = QM9XieGrossmanGraphs(
    #      species_list = species,
    #      edge_features = features['DistanceFeatures'],
    #      bond_angle_features = features['BondAngleFeatures'],
    #      dihedral_features = features['DihedralAngleFeatures'],
    #      node_feature_list = spec_features,
    #      alpha = features.get('alpha', 1.1)
    #  )

    #derivatives = QM9XieGrossmanDerivatives()

    #elif re.match('^simp', graph_type):

        #graphs = QM9SimpleXieGrossmanGraphs(
            #species_list = species, 
            #edge_features = features['DistanceFeatures'],
            #bond_angle_features = features['BondAngleFeatures'],
            #dihedral_features = features['DihedralAngleFeatures'],
            #node_feature_list = spec_features,
            #alpha = features.get('alpha', 1.1)
        #)

        #derivatives = QM9SimpleXieGrossmanDerivatives()


    #graphs = QM9ERH(
    #    species_list = species,
    #    distance_features = features['DistanceFeatures'],
    #    bond_angle_feature = features['BondAngleFeatures'],
    #    dihedral_angle_feature = features['DihedralAngleFeatures'],
    #    node_feature_list = spec_features,
    #    n_max_neighbours = n_max_neighbours,
    #)

    #derivatives = QM9ERHDerivatives(
    #    species_list = species,
    #    distance_features = features['DistanceFeatures'],
    #    bond_angle = features['BondAngleFeatures'],
    #    dihedral_angle = features['DihedralAngleFeatures'],
    #)

    graphs = XG(
        species_list=species,
        bond_angle_feature=bond_angle_feature,
        dihedral_angle_feature=dihedral_angle_feature,
        node_feature_list=spec_features,
        n_max_neighbours=n_max_neighbours,
    )

    #derivatives = XGDerivatives(
    #    species_list=species,
    #    distance_features=features['DistanceFeatures'],
    #    bond_angle=features['BondAngleFeatures'],
    #    dihedral_angle=features['DihedralAngleFeatures'],
    #)




    #elif re.match('^peri', graph_type):

        #graphs = PBCGraphs(
        #    species_list = species,
        #    edge_features = features['DistanceFeatures'],
        #    bond_angle_features = features['BondAngleFeatures'],
        #    dihedral_features = features['DihedralAngleFeatures'],
        #    node_feature_list = spec_features,
        #    n_max_neighbours = features.get('n_max_neighbours', 12),
        #    alpha = features.get('alpha', 1.1)
        #)

        #derivatives = None

    #else:  # we make this the default case

        #graphs = QM9ERH(
            #species_list = species, 
            #edge_feature = features['DistanceFeatures'],
            #bond_angle_feature = features['BondAngleFeatures'],
            #dihedral_angle_feature = features['DihedralAngleFeatures'],
            #node_feature_list = spec_features,
        #)

        #derivatives = QM9ERHDerivatives()



    return graphs
