from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

# Updated alerts list with fluorine included in the halogen pattern
# narrower
"""
alerts = [
    ("Alkyl esters of phosphonic or sulphonic acids", "COP(=O)(O)O"),
    ("Aromatic nitro groups", "[c][NX3](=O)=O"),
    ("Aromatic N-oxides", "[n+](=O)[O-]"),
    ("Aromatic mono- and dialkyl amino groups", "[c][NX3;H0,H1;!$(NC=O)]"),
    ("Alkyl hydrazines", "[NX3][NX3]"),
    ("Simple aldehyde", "[CX3H1](=O)[#6]"),
    ("N-methylol derivatives", "[NX3]CO"),
    ("Monohaloalkenes", "C=C[F,Cl,Br,I]"),
    ("S- or N- mustards", "N(CCCl)CCCl"),
    ("Acyl halides", "C(=O)Cl"),
    ("Propiolactones and propiosultones", "O=C1OCC1"),
    ("Epoxides and aziridines", "C1OC1"),
    ("Aliphatic halogens", "[CX4][F,Cl,Br,I]"),
    ("Alkyl nitrite", "CON=O"),
    ("Quinones", "O=C1C=CC(=O)C=C1"),
    ("N-nitroso", "[NX3][NX2]=O"),
    ("Aromatic amines and hydroxylamines", "[c][NX3H2,NX3H1]"),
    ("Azo, azoxy, diazo compounds", "N=N"),
    ("Alpha, beta unsaturated carbonyls", "[CX3]=[CX3][CX3](=O)[#6]"),
    ("Isocyanate and isothiocyanate groups", "N=C=O"),
    ("Alkyl carbamate and thiocarbamate", "OC(=O)N"),
    ("Heterocyclic/polycyclic aromatic hydrocarbons", "c1ccccc1"),
    ("Azide and triazene groups", "N=[NX1]=N"),
    ("Aromatic N-acyl amines", "[c][NX3][CX3](=O)"),
    ("Coumarins and Furocoumarins", "O=C1OC=CC2=CC=CC=C12"),
    ("Halogenated benzene", "[c][F,Cl,Br,I]"),
    ("Halogenated polycyclic aromatic hydrocarbon", "[c2][F,Cl,Br,I]"),
    ("Halogenated dibenzodioxins", "c1cc2Oc3c(cccc3Oc2cc1)[F,Cl,Br,I]"),
    ("Thiocarbonyls", "C(=S)"),
    ("Steroidal oestrogens", "C1CCC2C1(C)CCC3C2CCC4=CC(=O)CC=C34"),
    ("Trichloro/fluoro or tetrachloro/fluoro ethylene", "C([Cl,F])=C([Cl,F])[Cl,F]"),
    ("Pentachlorophenol", "c1(ccc(cc1Cl)Cl)Cl"),
    ("o-Phenylphenol", "c1ccccc1-c2ccccc2O"),
    ("Imidazole", "c1ncc[nH]1"),
    ("Dicarboximide", "O=C1NC(=O)C=C1"),
    ("Dimethylpyridine", "c1ccncc1C"),
    ("Michael acceptors", "C=CC=O"),
    ("Acrylamides", "C=CC(=O)N"),
    ("Alkylating sulfonates/mesylates/tosylates", "OS(=O)(=O)C"),
    ("Polyhalogenated alkanes", "C([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])"),
]
"""
# expanded
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
    ("Aliphatic halogens", "[CX4;!c][F,Cl,Br,I]"),
    ("Alkyl nitrite", "[CX4][OX2]N=O"),
    ("Quinones", "O=C1C=CC(=O)C=C1"),
    ("N-nitroso", "[NX3;H0,H1][NX2]=O"),
    ("Aromatic amines and hydroxylamines", "[c][NX3H2] or [c][NX3H1]O"),
    ("Azo, azoxy, diazo compounds", "[NX2]=[NX2] or [NX2]=N[O] or [NX2-]-[NX2+]"),
    ("Alpha, beta unsaturated carbonyls", "C=CC(=O)"),
    ("Isocyanate and isothiocyanate groups", "N=C=O or N=C=S"),
    ("Alkyl carbamate and thiocarbamate", "OC(=O)N or SC(=O)N"),
    ("Heterocyclic/polycyclic aromatic hydrocarbons", "c1ccccc1"),
    ("Azide and triazene groups", "N=[N+]=[N-] or N=N-N"),
    ("Aromatic N-acyl amines", "[c][NX3][CX3](=O)"),
    ("Coumarins and Furocoumarins", "O=C1OC=CC2=CC=CC=C12 or O=C1OC=CC2=CC=CC3=C21OCC3"),
    ("Halogenated benzene", "[c][F,Cl,Br,I]"),
    ("Halogenated polycyclic aromatic hydrocarbon", "[c1ccc2ccccc2c1][F,Cl,Br,I]"),
    ("Halogenated dibenzodioxins", "c1cc2Oc3c(cccc3Oc2cc1)[F,Cl,Br,I]"),
    ("Thiocarbonyls", "[CX3]=S"),
    ("Steroidal oestrogens", "C1CCC2C1(C)CCC3C2CCC4=CC(=O)CC=C34"),
    ("Trichloro/fluoro or tetrachloro/fluoro ethylene", "C([Cl,F])=C([Cl,F])[Cl,F]"),
    ("Pentachlorophenol", "c1(ccc(cc1Cl)Cl)Cl"),
    ("o-Phenylphenol", "c1ccccc1-c2ccccc2O"),
    ("Imidazole", "c1ncc[nH]1"),
    ("Dicarboximide", "O=C1NC(=O)C=C1"),
    ("Dimethylpyridine", "c1ccncc1C"),
    ("Michael acceptors", "C=CC=O"),
    ("Acrylamides", "C=CC(=O)N"),
    ("Alkylating sulfonates/mesylates/tosylates", "OS(=O)(=O)C"),
    ("Polyhalogenated alkanes", "C([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])([F,Cl,Br,I])"),
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
