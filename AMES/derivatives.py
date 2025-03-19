
from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import torch
from torch_geometric.data import Data, HeteroData

from device import device

class Derivatives(ABC):

    """

    :Class Derivatives:

    This is an Abstract Base Class (abc) whose purpose is to encapsulate and control
    the calculation of derivatives of the property being modelled with respect to input 
    parameters (atomic positions, strain tensor components
    or whatever). Since different types of graphs that we might use
    in the future may require specific strategies or expressions for the sought 
    derivatives, there will be a concrete child class of Derivatives for each type
    of graph/heterograph. 

    """

    def __init__(
        self,
        distance_features: Dict,
        bond_angle: bool = False,
        dihedral_angle: bool = False,
        **kwargs
    ) -> None:

        """
        Initialise the class; kwargs may be required in subclasses.

        Flags bond_angle and dihedral_angle indicate if bond angle and/or dihedral
        angle features are present in the edges of graphs 

        kwargs may be required in subclasses.
        """

        self.distance_features = distance_features
        self.bond_angle = bond_angle
        self.dihedral_angle = dihedral_angle
        self.kwargs = kwargs

    @abstractmethod
    def activate_derivatives(sample: Union[Data]) -> None:
        """
        In the input graph, set the gradient_required flag to True 
        for the required graph components. This will need to be done
        specifically in each subclass, so this method is abstract.

        Args:

        :param graph: Union[Data], the graph
           for which we need to set the appropriate derivative flags.

        :return None:
        """

        raise NotImplementedError

