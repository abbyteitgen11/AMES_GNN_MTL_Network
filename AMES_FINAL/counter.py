from pathlib import Path

import glob

from count_species import count_species

source_directory = '/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL/DataBase_AMES/FILES_XYZ'

source_directory_path = Path(source_directory)

input_file_ext = '.xyz'

pattern = '*' + input_file_ext

file_paths = list(source_directory_path.glob(pattern))

summary = count_species(file_paths)

print(summary) # Print out max atoms to determine n_neighbors

#print(max(summary.values()))

#print("Atom Type Summary:")
#for atom, count in summary.items():
#    print(f"{atom}: {count}")


