
from abc import ABC, abstractmethod
import re
from typing import Dict, List, Tuple

from mendeleev import element
import numpy as np

from exceptions import UnrecognizedKeyWord
from exceptions import UndefinedMandatoryObject
from exceptions import VariableOutOfBounds

"""
A class to define edge features dependent on distance; other
(angular) features are dealt with elsewhere.

"""

eps = 1.0e-8

class DistanceFeature(ABC):

    """

    Args:

    """

    def __init__(
        self,
        cutoff: float = 1.0,
        **kwargs
    ) -> None:

        self.cutoff = cutoff

    @abstractmethod
    def feature(self, r: float) -> float:

        """

        The distance feature 

        Args:
        
        :param: r: the value at which to evaluate the feature function.
        :type: float
        :return: feature value.
        :rtype: float

        """

        raise NotImplementedError

    @abstractmethod
    def derivative(self, r: float) -> float:

        """
        The distance feature derivative 

        Args:
        
        :param: r: the value at which to evaluate feature derivative.
        :type: float
        :return: feature derivative value.
        :rtype: float

        """

        raise NotImplementedError

    def cutoff(self) -> float:

        """
        Return the cutoff value.

        """

        return self.cutoff

class GaussianDistanceFeature(DistanceFeature):

    def __init__(
        self,
        cutoff: float = 1.0
    ) -> None:

        super().__init__(
            cutoff = cutoff  
        )

        # exponent is chosen so that Gaussian(cutoff) <= EPS ( 1.0e-8 defined above)

        self.exponent = -np.log( eps ) / ( self.cutoff * self.cutoff )

        self.cut_value = self.gaussian(self.cutoff)
        self.cut_dvalue = -2. * self.exponent * self.cutoff * self.cut_value

    def gaussian(self, r: float) -> float:

        arg = -self.exponent * r * r

        return np.exp( arg )

    def feature(self, r: float) -> float:

        result = self.gaussian(r) - self.cut_value - \
                   self.cut_dvalue * ( r - self.cutoff )

        return result

    def derivative(self, r: float) -> float:

        result = -2. * self.exponent * r * self.gaussian(r) - self.cut_dvalue

        return result

class FermiDistanceFeature(DistanceFeature):

    def __init__(
        self,
        r_fermi: float = 1.0,
        delta: float = 0.5,
        precision: float = 7.
    ) -> None:

       self.r_fermi = r_fermi
       self.precision = precision

       cutoff = self.r_fermi + delta

       super().__init__(
           cutoff = cutoff
       )

       self.alpha = delta / ( self.precision * np.log(10.) )

    def feature(self, r: float) -> float:

       arg = (r - self.r_fermi)/self.alpha

       value = 1. / (np.exp(arg) + 1.)

       return value

    def derivative(self, r: float) -> float:

       arg = (r - self.r_fermi)/self.alpha

       value = np.exp(arg)

       result = -value / (value + 1.) / (value + 1.) / self.alpha

       return result

def set_up_distance_features(
       species: List = ['N', 'C', 'H', 'O', 'S', 'Cl', 'Be', 'Br',
                        'Pt', 'P', 'F', 'As', 'Hg', 'Zn', 'Si', 'V',
                        'I', 'B', 'Sn', 'Ge', 'Ag', 'Sb', 'Cu', 'Cr',
                        'Pb', 'Mo', 'Se', 'Al', 'Cd', 'Mn', 'Fe', 'Ga',
                        'Pd', 'Na', 'Ti', 'Bi', 'Co', 'Ni', 'Ce', 'Ba', 'Zr', 'Rh'],
       delta: float = 0.2,
       precision: float = 7.
    ) -> Dict:

    """
    Given a list of species, and a feature type, it returns a dictionary of
    the features, where each key is made up of a tuple of species, and the
    value returned is the feature object for that particular pair of species.

    """

    # features = GaussianDistanceFeature( cutoff = cutoff, 
    #                                     exponent = kwargs['exponent'] )

    elements = []

    for spec in species:
        elements.append(element(spec))

    features = {}

    for s1, spec1 in enumerate(species):
        for s2, spec2 in enumerate(species):

            key = (spec1, spec2)

            cutoff = (elements[s1].covalent_radius + \
                            elements[s2].covalent_radius) / 100. 

            # the factor 1/100. is because Mendeleev stores radii in pm

            # value = GaussianDistanceFeature(cutoff)

            value = FermiDistanceFeature(r_fermi = cutoff, delta = delta,
                                         precision = precision)

            features[key] = value

    return features

def set_up_features(input_data: Dict) -> Dict:

    """

    This function gets passed the input data, searches for the
    definitions of edge, bond angle and dihedral angle feature instances, 
    and returns them in the form of a dictionary

    Args:

    :param: input_data 
    :type: dict
    :return: a Dictionary of features for edges, bond angles and dihedral angles 
             in the case of standard graphs, or ee_edge_features, ei_edge_features
             etc in the case of heterographs
    :rtype: Dict[Features]

    """

    features_dict = {}

    # first edge features (features depending only on bond-distance )

    edge_features = input_data.get("EdgeDistanceFeatures", True)

    if edge_features:

        species = input_data.get('species', ['N', 'C', 'H', 'O', 'S', 'Cl', 'Be', 'Br', 'Pt', 'P', 
                    'F', 'As', 'Hg', 'Zn', 'Si', 'V', 'I', 'B', 'Sn', 'Ge', 
                    'Ag', 'Sb', 'Cu', 'Cr', 'Pb', 'Mo', 'Se', 'Al', 'Cd', 
                    'Mn', 'Fe', 'Ga', 'Pd', 'Na', 'Ti', 'Bi', 'Co', 'Ni', 
                    'Ce', 'Ba', 'Zr', 'Rh'])

        edges = set_up_distance_features(species)

        features_dict['DistanceFeatures'] = edges
 
    else:

      raise UndefinedMandatoryObject('edge_features')

    # bond-angle features

    angle_features = input_data.get('BondAngleFeatures', False)
    features_dict['BondAngleFeatures'] = angle_features

    # finally, dihedral angle features, if given (if not, return as None)

    dihedral_features = input_data.get('DihedralAngleFeatures', False)
    features_dict['DihedralAngleFeatures'] = dihedral_features

    return features_dict
