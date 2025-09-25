from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# Updated alerts list with fluorine included in the halogen pattern
# narrower
alerts = [
        ("Alkyl esters of phosphonic or sulphonic acids", "C[OX2]P(=O)(O)O or C[OX2]S(=O)(=O)O"),
        ("Aromatic nitro groups", "[c][NX3](=O)=O"),
        ("Aromatic N-oxides", "[n+](=O)[O-]"),
        ("Aromatic mono- and dialkyl amino groups", "[c][NX3;H0,H1;!$(NC=O)]"),
        ("Alkyl hydrazines", "[NX3][NX3]"),
        ("Simple aldehyde", "[CX3H1](=O)[#6]"),
        ("N-methylol derivatives", "[NX3]CO"),
        ("Monohaloalkenes", "C=C[F,Cl,Br,I]"),
        ("S- or N- mustards", "N(CCCl)CCCl or S(CCCl)CCCl"),
        ("Acyl halides", "[CX3](=O)[F,Cl,Br,I]"),
        ("Propiolactones and propiosultones", "O=C1OCC1 or O=S1OCC1"),
        ("Epoxides and aziridines", "C1OC1 or C1NC1"),
        ("Heterocyclic/polycyclic aromatic hydrocarbons", "c1ccccc1"),
        ("Aliphatic halogens", "[CX4;!c][F,Cl,Br,I]"),
        ("Alkyl nitrite", "[CX4][OX2]N=O"),
        ("Quinones", "O=C1C=CC(=O)C=C1"),
        ("N-nitroso", "[NX3;H0,H1][NX2]=O"),
        ("Aromatic amines and hydroxylamines", "[c][NX3H2] or [c][NX3H1]O"),
        ("Azo, azoxy, diazo compounds", "[NX2]=[NX2] or [NX2]=N[O] or [NX2-]-[NX2+]"),
        ("Alpha, beta unsaturated carbonyls", "C=CC(=O)"),
        ("Isocyanate and isothiocyanate groups", "N=C=O or N=C=S"),
        ("Alkyl carbamate and thiocarbamate", "OC(=O)N or SC(=O)N"),
        ("Azide and triazene groups", "N=[N+]=[N-] or N=N-N"),
        ("Aromatic N-acyl amines", "[c][NX3][CX3](=O)"),
        ("Coumarins and Furocoumarins", "O=C1OC=CC2=CC=CC=C12 or O=C1OC=CC2=CC=CC3=C21OCC3"),
        ("Michael acceptors", "C=CC=O"),
        ("Acrylamides", "C=CC(=O)N"),
        ("Alkylating sulfonates/mesylates/tosylates", "OS(=O)(=O)C"),
    ]

# Convert to RDKit mols
mols = []
names = []
for name, smi in alerts:
    mol = Chem.MolFromSmarts(smi) or Chem.MolFromSmiles(smi)
    if mol:
        mols.append(mol)
        names.append(name)
    else:
        print(f"Could not parse: {name} -> {smi}")

# Draw grid image
img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(250, 250), legends=names, useSVG=False)

# Show with matplotlib
plt.figure(figsize=(20, 20))
plt.imshow(img)
plt.axis("off")
plt.show()
