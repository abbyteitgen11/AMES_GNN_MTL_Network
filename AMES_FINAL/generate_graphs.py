
from collections.abc import Generator
from pathlib import Path
import pickle

from torch_geometric.data import Data

from atomic_structure_graphs import AtomicStructureGraphs

def generate_graphs(graphs: AtomicStructureGraphs,
        files: Generator[Path],
        target_dir: Path,
        set: tuple,
        output_file_ext: str = '.pkl'
        ) -> None:

    """
    This function uses an object of type AtomicStructureGraphs to 
    create graphs (instances of pytorch_geometric.data.Data) from molecular
    information contained in the given files (typically in xyz format) and 
    stores the resulting graphs as pickled files in the target directory
  
    Args:

    :param graphs: an instance of AtomicStructureGraphs used to transform molecular
                   structural information into a PyG Data object (a graph) 
    :param files: a Generator object yielding the the paths of files containing 
                   the molecular information for a given data-base.
    :param target_dir: the directory path where to place the pickled graphs

    """

    for file in files:

        words = file.parts

        input_file = words[-1]

        words = input_file.split('.')
 
        graph_file = str(target_dir) + '/' + words[0] + output_file_ext

        file_name = str(file)

        structure_graph = graphs.structure2graph(file_name, set)

        with open( graph_file, 'wb' ) as outfile:
           pickle.dump(structure_graph, outfile)
