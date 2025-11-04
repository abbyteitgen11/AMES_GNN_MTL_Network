from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt

alerts = [
        ("Acyl halides", "[Br,Cl,F,I][CX3](=[OX1])[#1,*&!$([OH1])&!$([SH1])]"),
        ("Alkyl and aryl N-nitroso groups", "[#6][NX3][NX2]=[OX1]"),
        ("Alkyl or benzyl esters of phosphonic or sulphonic acids","[$([Sv6X4;!$([Sv6X4][OH]);!$([Sv6X4][SH]);!$([Sv6X4][O-]);!$([Sv6X4][S-])](=[OX1])(=[OX1])[$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH0](C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH2]c1ccccc1)]),$([Pv5X4;!$([Pv5X4][OH]);!$([Pv5X4][SH]);!$([Pv5X4][O-]);!$([Pv5X4][S-])](=[OX1])([$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH0](C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH2]c1ccccc1)])[$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH0](C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2]C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])(C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I])C([#1,F,Cl,Br,I])([#1,F,Cl,Br,I])[#1,F,Cl,Br,I]),$([OX2][CH2]c1ccccc1)])]"),
        ("Alkyl carbamate and thiocarbamates", "[NX3]([C,#1])([C,#1])[CX3](=[OX1,Sv2X1])[OX2,Sv2X2]C"),
        ("Hydrazines", "[NX3;!$([NX3](=[OX1])=[OX1]);!$([NX3+](=[OX1])[O-])][NX3;!$([NX3](=[OX1])=[OX1]);!$([NX3+](=[OX1])[O-])]"),
        ("Alkyl nitrites", "[OX1]=[NX2][OX2][CX4]"),
        ("Aliphatic azo and azoxy groups", "[$([C,#1][NX2]=[NX2][C,#1]),$([CX3]=[NX2+]=N),$([CX3]=[NX2+]=[NX1-]),$([CX3-][NX2+]#[NX1]),$([CX3][NX2]#[NX1]),$(C[NX2]=N(=O)[*]),$(C[NX2]=[N+]([O-])[*])]"),
        ("Aliphatic halogens", "[CX4;!H0][Br,Cl,I]"),
        ("Aliphatic N-nitro groups", "[NX3]([#1,C])([#1,C])[$([NX3+](=[OX1])[O-]),$([NX3](=O)=O)]"),
        ("Alpha, beta unsaturated aliphatic alkoxy groups", "C[CX3;H1]=[CX3;H1][OX2][#6]"),
        ("Alpha, beta unsaturated carbonyls", " [CX3]([!$([OH]);!$([O-])])(=[OX1])[CX3H1]=[CX3]([$([CH3]),$([CH2][CH3]),$([CH2][CH2][CH3]),$([CH]([CH3])[CH3]),$([CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH3]),$([CH2][CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH3]),$([CH2][CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH2][CH3]),$([CH2][CH]([CH3])[CH2][CH3]),$([CH2][CH2][CH]([CH3])[CH3]),$([CH]([CH2][CH3])[CH2][CH3]),$([CH]([CH3])[CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH2][CH3]),$([CH2][CH0]([CH3])([CH3])[CH3]),$([#1,#7,#8,F,Cl,Br,I,#15,#16,#5]),$([CH]=[CH][#6]);!$([a!r0])])[$([CH3]),$([CH2][CH3]),$([CH2][CH2][CH3]),$([CH]([CH3])[CH3]),$([CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH3]),$([CH2][CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH3]),$([CH2][CH2][CH2][CH2][CH3]),$([CH]([CH3])[CH2][CH2][CH3]),$([CH2][CH]([CH3])[CH2][CH3]),$([CH2][CH2][CH]([CH3])[CH3]),$([CH]([CH2][CH3])[CH2][CH3]),$([CH]([CH3])[CH]([CH3])[CH3]),$([CH0]([CH3])([CH3])[CH2][CH3]),$([CH2][CH0]([CH3])([CH3])[CH3]),$([#1,#7,#8,F,Cl,Br,I,#15,#16,#5]),$([CH]=[CH][#6]);!$([a!r0])]"),
        ("Aromatic amines and hydroxylamines", "[a!r0][$([NH2]),$([NX3][OX2H1]),$([NX3][OX2][CX3H1](=[OX1])),$([NX2]=[CH2]),$([NX2]=C=[OX1]);!$([NX3,NX2]a(a-[!#1])a-[!#1]);!$([NX3,NX2]aa-C(=[OX1])[OH]);!$([NX3,NX2]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3,NX2]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3,NX2]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])]"),
        ("Aromatic diazo compounds", "[$([NX2]([a!r0])=[NX2][a!r0]);!$([NX2](aaS(=O)(=O)[OH])=[NX2]aaS(=O)(=O)[OH]);!$([NX2](aaS(=O)(=O)[OH])=[NX2]aaaS(=O)(=O)[OH]);!$([NX2](aaaS(=O)(=O)[OH])=[NX2]aaS(=O)(=O)[OH]);!$([NX2](aaS(=O)(=O)[OH])=[NX2]aaaaS(=O)(=O)[OH])!$([NX2](aaaaS(=O)(=O)[OH])=[NX2]aaS(=O)(=O)[OH]);!$([NX2](aaaS(=O)(=O)[OH])=[NX2]aaaS(=O)(=O)[OH]);!$([NX2](aaaS(=O)(=O)[OH])=[NX2]aaaaS(=O)(=O)[OH]);!$([NX2](aaaaS(=O)(=O)[OH])=[NX2]aaaS(=O)(=O)[OH]);!$([NX2](aaaaS(=O)(=O)[OH])=[NX2]aaaaS(=O)(=O)[OH])]"),
        ("Aromatic nitro groups", "[a!r0][$([NX3+](=[OX1])[O-]),$([NX3](=[OX1])=[OX1]);!$([NX3]a(a-[!#1])a-[!#1]);!$([NX3]aa-C(=[OX1])[OH]);!$([NX3]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])]"),
        ("Aromatic nitroso groups", "[a!r0][NX2]=[OX1]"),
        ("Aromatic mono- and dialkyl amino groups", "[a!r0][$([NX3;H1][CH3]),$([NX3;H1][CH2][CH3]),$([NX3]([CH3])[CH3]),$([NX3]([CH3])[CH2][CH3]),$([NX3]([CH2][CH3])[CH2][CH3]);!$([NX3]a(a-[!#1])a-[!#1]);!$([NX3]aa-C(=[OX1])[OH]);!$([NX3]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])]"),
        ("Aromatic N-acyl amines", "[a!r0][$([NX3;H1]),$([NX3][CH3]);!$([NX3]a(a-[!#1])a-[!#1]);!$([NX3]aa-C(=[OX1])[OH]);!$([NX3]aa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaa-[Sv6X4](=[OX1])(=[OX1])[OH]);!$([NX3]aaaa-[Sv6X4](=[OX1])(=[OX1])[OH])][CX3](=[OX1])([$([#1]),$([CH3])])"),
        ("Aromatic N-oxides", "[O-][N+]1=CC=CC=C1"),
        ("Azide and triazene groups", "[$([NX2!R]=[NX2!R][NX3!R]),$([NX2]=[NX2+]=[NX1-]),$([NX2]=[NX2+]=N)]"),
        ("Coumarins and Furocoumarins", "[$(c1cccc2c1oc(=O)cc2),$(C1=CC(=O)OC2=CC=CC=C12),$(C1=CC(=O)Oc2ccccc12)]"),
        ("Epoxides and aziridines", "[CX4]1[OX2,NX3][CX4]1"),
        ("Polycyclic aromatic hydrocarbons", "[$([cX3R3]),$([cX3;R1,R2,R3]1[cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R2,R3]2[cX3;R2,R3]1[cX3;R1,R2,R3][cX3;R2,R3]3[cX3;R2,R3]([cX3;R1,R2,R3]2)[cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R1,R2,R3][cX3;R1,R2,R3]3)].[!$([n,o,s])]"),
        ("Heterocyclic polycyclic aromatic hydrocarbons", "[$([aR3].[n,o,s]),$([$([aR2]([aR])([aR])([aR]))].[$([aR2]([aR])([aR])([aR]))].[$([aR2]([aR])([aR])([aR]))].[$([aR2]([aR])([aR])([aR]))].[n,o,s])]"),
        ("Isocyanate and isothiocyanate groups", "[NX2]=[CX2]=[OX1,Sv2X1]"),
        ("Monohaloalkenes", "[CX3]([CX4,#1])([F,Cl,Br,I])=[CX3]([CX4,#1])[!F!Cl!Br!I]"),
        ("N-methylol derivatives", "[OX2;H1][CH2][NX3]"),
        ("Propiolactones and propiosultones", "[$([OX2]1[CX4][CX4][CX3]1(=[OX1])),$([CX4]1[CX4][CX4][Sv6;X4](=[OX1])(=[OX1])[OX2]1)]"),
        ("Quinones", "[$([#6X3]1=,:[#6X3]-,:[#6X3](=[OX1])-,:[#6X3]=,:[#6X3]-,:[#6X3]1(=[OX1])),$([#6X3]1(=[OX1])-,:[#6X3](=[OX1])-,:[#6X3]=,:[#6X3]-,:[#6X3]=,:[#6X3]1)]"),
        ("Simple aldehydes", "[CX3]([H])(=[OX1])[#1,#6&!$([CX3]=[CX3])]"),
        ("S- or N- mustards", "[F,Cl,Br,I][CH2][CH2][NX3,SX2][CH2][CH2][F,Cl,Br,I]")
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
