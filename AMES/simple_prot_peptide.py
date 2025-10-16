import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
import os

def extract_chain_sequence(pdb_path, chain_id):
    """Extracts 1-letter amino acid sequence for a given chain in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("struct", pdb_path)
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [residue for residue in chain if residue.get_id()[0] == ' ']
                seq = "".join([seq1(res.get_resname()) for res in residues])
                return seq
    return None


# Load the space-separated text file
df = pd.read_csv(
    "/Users/abigailteitgen/Downloads/pepbdb-20200318/peptidelist.txt",
    delim_whitespace=True,
    header=None,
    names=[
        "pdb_id", "pep_chain", "pep_len", "pep_atoms",
        "prot_chain", "prot_atoms", "num_contacts",
        "unknown1", "unknown2", "resolution", "mol_type"
    ]
)

print(df.head())

protein_seqs, peptide_seqs, valid_rows = [], [], []

for i, row in df.iterrows():
    pdb_folder = os.path.join("pepbdb", row.pdb_id.lower())
    pdb_file = os.path.join(pdb_folder, f"{row.pdb_id.lower()}.pdb")

    if not os.path.exists(pdb_file):
        continue  # skip missing files

    pep_seq = extract_chain_sequence(pdb_file, row.pep_chain)
    prot_seq = extract_chain_sequence(pdb_file, row.prot_chain)

    if pep_seq and prot_seq and len(pep_seq) > 0 and len(prot_seq) > 0:
        peptide_seqs.append(pep_seq)
        protein_seqs.append(prot_seq)
        valid_rows.append(row)

seq_df = pd.DataFrame(valid_rows)
seq_df["peptide_seq"] = peptide_seqs
seq_df["protein_seq"] = protein_seqs


AA = "ACDEFGHIKLMNPQRSTVWY"

def aa_composition(seq):
    seq = seq.upper()
    counts = np.array([seq.count(a) for a in AA])
    return counts / len(seq)

def make_features(protein_seq, peptide_seq):
    return np.concatenate([aa_composition(protein_seq), aa_composition(peptide_seq)])

# Build features and targets
X = np.array([make_features(p, q) for p, q in zip(seq_df["protein_seq"], seq_df["peptide_seq"])])
y = seq_df["num_contacts"].astype(float).values

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
