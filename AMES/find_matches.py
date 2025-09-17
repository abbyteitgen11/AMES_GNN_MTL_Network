from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import pandas as pd

# Create fingerprints
def mol_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)

# Assume you have a DataFrame "df_mols" with columns: mol_id, smiles, label, prediction, alert_present
df_mols = df.drop_duplicates("mol_id")[["mol_id", "smiles", "label", "prediction", "alert_present"]]

fps = {row.mol_id: mol_to_fp(row.smiles) for row in df_mols.itertuples() if mol_to_fp(row.smiles) is not None}

# Now search for pairs: one with alert, one without
pairs = []
for id1, fp1 in fps.items():
    for id2, fp2 in fps.items():
        if id1 >= id2:  # avoid repeats
            continue
        sim = DataStructs.TanimotoSimilarity(fp1, fp2)

        mol1 = df_mols[df_mols.mol_id == id1].iloc[0]
        mol2 = df_mols[df_mols.mol_id == id2].iloc[0]

        # Check criteria: one has alert & toxic, other no alert & non-toxic
        if (
            mol1.alert_present and mol1.prediction == 1 and mol1.label == 1 and
            not mol2.alert_present and mol2.prediction == 0 and mol2.label == 0 and
            sim > 0.7  # similarity threshold
        ):
            pairs.append((id1, id2, sim))

print("Example pairs found:", pairs[:5])
