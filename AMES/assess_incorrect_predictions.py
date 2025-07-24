import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.stats import ttest_ind

# Load the dataset
#df = pd.read_csv("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/data.csv")

# Filter for 'External' partition
#external_df = df[df["Partition"] == "External"]

# Select only the SMILES column (you can also keep IDs if needed)
#external_smiles = external_df[["SMILES RDKit"]].dropna().reset_index(drop=True)

# Optionally rename the column
#external_smiles = external_smiles.rename(columns={"SMILES RDKit": "SMILES"})

# Save to CSV
#external_smiles.to_csv("external_smiles.csv", index=False)

#print(f"Saved {len(external_smiles)} SMILES strings to external_smiles.csv")



# Load your data (must contain 'SMILES' and 'IsMisclassified' columns)
df = pd.read_csv("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/AMES/external_smiles.csv")

# Filter valid SMILES
df["Mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
df = df[df["Mol"].notnull()]

# Compute descriptors
df["MolWt"] = df["Mol"].apply(Descriptors.MolWt)
df["NumAromaticRings"] = df["Mol"].apply(Descriptors.NumAromaticRings)
df["NumHDonors"] = df["Mol"].apply(Descriptors.NumHDonors)
df["NumHAcceptors"] = df["Mol"].apply(Descriptors.NumHAcceptors)
df["NumRotatableBonds"] = df["Mol"].apply(Descriptors.NumRotatableBonds)
df["TPSA"] = df["Mol"].apply(Descriptors.TPSA)

# Separate into two groups
mis = df[df["IsMisclassified"] == True]
correct = df[df["IsMisclassified"] == False]

# Descriptor list
descriptor_cols = ["MolWt", "NumAromaticRings", "NumHDonors", "NumHAcceptors", "NumRotatableBonds", "TPSA"]

# 1. Plot descriptor distributions
plot_data = []
labels = []
for col in descriptor_cols:
    plot_data.extend([mis[col], correct[col]])
    labels.extend([f"{col}\nMis", f"{col}\nCorrect"])

plt.figure(figsize=(16, 6))
plt.boxplot(plot_data, labels=labels, patch_artist=True)
plt.ylim(0, 1000)
plt.xticks(rotation=45)
plt.title("Descriptor Comparison: Misclassified vs Correct")
plt.ylabel("Value")
plt.tight_layout()
plt.show()

# 2. Run t-tests and print results
print("T-test Results:\n")
for col in descriptor_cols:
    t_stat, p_val = ttest_ind(mis[col], correct[col], equal_var=False)
    print(f"{col}:")
    print(f"  Mis Mean = {mis[col].mean():.2f}, Correct Mean = {correct[col].mean():.2f}")
    print(f"  t = {t_stat:.3f}, p = {p_val:.3e}")
    print()
