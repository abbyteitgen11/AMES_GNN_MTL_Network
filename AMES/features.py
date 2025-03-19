import re
from typing import Dict, Tuple

import numpy as np

from exceptions import UnrecognizedKeyWord
from exceptions import UndefinedMandatoryObject
from exceptions import VariableOutOfBounds

r"""
A general class to define features, which can be distance, or
angle-based, or general. Their general expression is adopted
from Xie & Grossman

.. math::
      u_k(x) = \exp\left(-(x - \mu_k)^2/\sigma^2)

where the n_features :math:`\mu_k` points are evenly distributed
between x_min and x_max

Also provided a function that returns instances of different types
of features (edges, bond-angle, dihedral-angle)

"""

class Features:

    r"""

    Args:

    :param: x_min: the lower limit of the argument range
    :type: float
    :param: x_max: the upper limit
    :type: float
    :param: n_features: total number of feature values
    :type: int
    :param: sigma: the Gaussian inverse exponent of the features (see above)
    :type: float
    :param: norm: if True, the features are normalised in the
           following sense:
           .. math::
              \sum_k^n_features u_k(x) = 1
    :type: bool

    """

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 1.0,
        n_features: int = 40,
        sigma: float = 0.2,
        norm: bool = False,
    ) -> None:

        self._x_min = x_min
        self._x_max = x_max
        self._n_features = n_features
        self._sigma = sigma
        self._norm = norm

        self.points = np.linspace(self._x_min, self._x_max, self._n_features)

    def u_k(self, x: float) -> np.ndarray:

        r"""

        The edge feature evaluated at all edge feature points.

        Args:
        
        :param: x: the value at which to evaluate features, .. math:: x \in [x_min,x_max]
        :type: float
        :return: array of feature values.
        :rtype: np.ndarray

        """

        x_val = np.around(x,6)

        if x_val < self._x_min or x_val > self._x_max:
            raise VariableOutOfBounds('Value {} out of bounds.'.format(x))

        val = (x - self.points) / self._sigma
        val2 = -val * val

        feature = np.exp(val2)

        if self._norm:
            norm = np.sum(feature)
            if norm > 0.0:
                feature /= norm

        return feature

    def du_k(self, x: float) -> np.ndarray:

        """
        The edge feature derivative evaluated at all edge feature points.

        Args:
        
        :param: x: the value at which to evaluate feature derivatives.
        :type: float
        :return: array of feature derivative values.
        :rtype: np.ndarray

        """

        if x < self._x_min or x > self._x_max:
            raise VariableOutOfBounds('Value {} out of bounds.'.format(x))

        val = (x - self.points) / self._sigma
        val2 = -val * val

        du = -2.0 * val * np.exp(val2) / self._sigma

        return du

    def parameters(self) -> Dict:

        """
        Interrogate the instance about its defining parameters.

        :return: dictionary of feature object parameters.
        :rtype: dict

        """

        return {
           "x_min": self._x_min,
           "x_max": self._x_max,
           "n_features": self._n_features,
           "sigma": self._sigma,
           "norm": self._norm,
        }

    def n_features(self) -> int:
        """
        Return number of features.

        :return: number of features.
        :rtype: int
        """
        return self._n_features

def set_up_features(input_data: Dict) -> Dict:

    """

    This function gets passed the input data, searches for the
    definitions of edge, bond angle and dihedral angle feature instances, 
    and returns them in the form of a dictionary

    Args:

    :param: input_data 
    :type: dict
    :return: a Dictionary of features for edges, bond angles and dihedral angles 
          
    :rtype: Dict[Features]

    """

    features_dict = {}

    # first edge features (features depending only on bond-distance )
       

    edge_features = input_data.get("EdgeFeatures", None)

    if edge_features is not None:

        r_min = edge_features.get('r_min', 0.0)
        r_max = edge_features.get('r_max', 8.0)
        n_edge_features = edge_features.get('n_features', 10)
        sigma = edge_features.get('sigma', 0.2)
        norm = edge_features.get('norm', False)

        edges = Features(
            x_min=r_min, x_max=r_max, n_features=n_edge_features,
            sigma=sigma, norm=norm
        )

        features_dict['edge_features'] = edges
 
    else:

      raise UndefinedMandatoryObject('edge_features')

    # bond-angle features

    angle_features = input_data.get('AngleFeatures', None)

    if angle_features is not None:

        theta_min = angle_features.get('theta_min', -1.0)
        theta_max = angle_features.get('theta_max', 1.0)
        n_angle = angle_features.get('n_features', 10)
        sigma = angle_features.get('sigma', 0.1)
        norm = angle_features.get('norm', False)

        bond_angle = Features(
            x_min=theta_min, x_max=theta_max, n_features=n_angle,
            sigma=sigma, norm=norm
        )

        features_dict['bond_angle_features'] = bond_angle

    else:

        features_dict['bond_angle_features'] = None

    # finally, dihedral angle features, if given (if not, return as None)

    dihedral_features = input_data.get('DihedralFeatures', None)

    if dihedral_features is not None:

        theta_min = dihedral_features.get('theta_min', -1.0)
        theta_max = dihedral_features.get('theta_max', 1.0)
        n_dihedral = angle_features.get('n_features', 10)
        sigma = angle_features.get('sigma', 0.1)
        norm = angle_features.get('norm', False)

        dihedral_angle = Features(
            x_min=theta_min, x_max=theta_max, n_features=n_dihedral,
            sigma=sigma, norm=norm
        )
       
        features_dict['dihedral_angle_features'] = dihedral_angle

    else:

        features_dict['dihedral_angle_features'] = None

    return features_dict
