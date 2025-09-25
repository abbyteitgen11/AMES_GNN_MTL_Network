import pandas as pd

# Load the spreadsheet
file_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/compare_GNN_paper_ISSSTY.xlsx"  # change to your actual file path
df = pd.read_excel(file_path)

# Filter: keep only rows where smiles_paper is NOT in smiles_isssty
filtered_df = df[~df["smiles_paper"].isin(df["smiles_isssty"])]

# Keep only the first three columns
result = filtered_df[["smiles_paper", "ames_paper", "source_paper"]]

# Save the result to a new Excel file
output_path = "/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/final_additional_dataset.xlsx"
result.to_excel(output_path, index=False)

print(f"Filtered file saved as: {output_path}")
