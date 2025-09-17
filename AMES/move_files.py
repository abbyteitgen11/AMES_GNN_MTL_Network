import os
import shutil
import pandas as pd


# Paths (set these to your actual folders)
subset_file = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/eval_paper_results.xlsx"
source_dir = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/GraphDataBase_AMES/test"
dest_dir = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/GraphDataBase_AMES/dev/fold_5/test"

# Make destination dir if it doesn’t exist
#os.makedirs(dest_dir, exist_ok=True)

# Load subset spreadsheet
df = pd.read_excel('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/eval_paper_results.xlsx', sheet_name='Indices')

# Loop through Index values
for idx in df["Fold_5"]:
    file_num = idx + 1
    if pd.isna(file_num):
        break
    file_num_int = int(file_num)
    prefix = f"{file_num_int}_"

    # Look for a file that starts with "<file_num>_"
    matches = [f for f in os.listdir(source_dir) if f.startswith(prefix) and f.endswith(".pkl")]

    if matches:
        # If multiple, just take the first one
        filename = matches[0]
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(dest_dir, filename)
        shutil.copy(src_path, dst_path)
    else:
        print(f"WARNING: no file found for Index {idx} (looking for prefix {prefix})")

print("Copy complete.")
