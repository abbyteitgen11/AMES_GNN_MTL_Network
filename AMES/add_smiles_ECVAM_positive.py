#!/usr/bin/env python3
"""
Augment ECVAM_positive.xls with SMILES using online lookups.

- Tries PubChem (PUG-REST) by CAS → then by Name
- Falls back to NIH Cactus CIR by CAS/Name
- Falls back to OPSIN by Name
- Writes: ECVAM_positive_with_smiles.xlsx

Requirements:
    pip install requests openpyxl xlrd
Optional (for salt/mixture cleanup):
    pip install rdkit-pypi
"""

import re
import time
import json
import urllib.parse
from pathlib import Path

import pandas as pd
import requests

# ----------------- Config -----------------
IN_XLS  = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/ECVAM_positive_simple.xlsx")   # columns: 'CAS No.' , 'Ames Overall', and (maybe) a name column
OUT_XLSX = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/ECVAM_positive_simple_with_smiles.xlsx")

# Rate limit (be kind to public APIs)
SLEEP_BETWEEN_CALLS = 0.2  # seconds

# If you want to drop counterions (e.g., "." fragments) keep the largest fragment:
USE_RDKIT_LARGEST_FRAGMENT = True  # set True if RDKit installed

CAS_REGEX = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

# If your ECVAM file uses a different name column, add it here (we’ll take the first one that exists).
POSSIBLE_NAME_COLS = [
    "Chemical"
]

# ------------- Optional: RDKit helpers -------------
def largest_fragment_smiles(smiles: str) -> str:
    if not USE_RDKIT_LARGEST_FRAGMENT:
        return smiles
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return smiles
        frags_sorted = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
        return Chem.MolToSmiles(frags_sorted[0], isomericSmiles=True)
    except Exception:
        return smiles

# ------------- Extractors -------------
def extract_first_cas(cell) -> str | None:
    if pd.isna(cell):
        return None
    s = str(cell).replace(" 00:00:00", "")
    m = CAS_REGEX.search(s)
    return m.group(0) if m else None

# ------------- Query helpers -------------
def pubchem_props_by_name(name_or_cas: str) -> dict | None:
    """
    PubChem PUG REST: compound/name/{identifier}/property/IsomericSMILES,CanonicalSMILES,InChIKey/JSON
    Using CAS as 'name' usually works.
    """
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    props = "property/IsomericSMILES,CanonicalSMILES,InChIKey/JSON"
    url = base + urllib.parse.quote(name_or_cas) + "/" + props
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    try:
        data = r.json()
        recs = data.get("PropertyTable", {}).get("Properties", [])
        return recs[0] if recs else None
    except json.JSONDecodeError:
        return None

def cir_smiles(identifier: str) -> str | None:
    """
    NIH Cactus CIR: /chemical/structure/{identifier}/smiles
    """
    base = "https://cactus.nci.nih.gov/chemical/structure/"
    url = base + urllib.parse.quote(identifier) + "/smiles"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    txt = r.text.strip()
    return txt if txt and "Not Found" not in txt else None

def opsin_smiles(name: str) -> str | None:
    """
    OPSIN: /opsin/{name}.smiles
    """
    base = "https://opsin.ch.cam.ac.uk/opsin/"
    url = base + urllib.parse.quote(name) + ".smiles"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return None
    txt = r.text.strip()
    return txt if txt and "OPSIN-Error" not in txt else None

# ------------- Main lookup logic -------------
def lookup_smiles(cas: str | None, name: str | None) -> tuple[str | None, str]:
    """
    Returns (SMILES, SourceTag)
    Priority:
      1) PubChem by CAS
      2) PubChem by Name
      3) CIR by CAS
      4) CIR by Name
      5) OPSIN by Name
    """
    # PubChem by CAS
    if cas:
        try:
            rec = pubchem_props_by_name(cas)
            if rec:
                smi = rec.get("IsomericSMILES") or rec.get("CanonicalSMILES")
                if smi:
                    return largest_fragment_smiles(smi), "PubChem (CAS)"
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN_CALLS)

    # PubChem by Name
    if name:
        try:
            rec = pubchem_props_by_name(name)
            if rec:
                smi = rec.get("IsomericSMILES") or rec.get("CanonicalSMILES")
                if smi:
                    return largest_fragment_smiles(smi), "PubChem (Name)"
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN_CALLS)

    # CIR by CAS
    if cas:
        try:
            smi = cir_smiles(cas)
            if smi:
                return largest_fragment_smiles(smi), "CIR (CAS)"
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN_CALLS)

    # CIR by Name
    if name:
        try:
            smi = cir_smiles(name)
            if smi:
                return largest_fragment_smiles(smi), "CIR (Name)"
        except Exception:
            pass
        time.sleep(SLEEP_BETWEEN_CALLS)

    # OPSIN by Name (IUPAC-style names)
    if name:
        try:
            smi = opsin_smiles(name)
            if smi:
                return largest_fragment_smiles(smi), "OPSIN (Name)"
        except Exception:
            pass

    return None, ""

def main():
    # Load ECVAM positive (.xls)
    df = pd.read_excel(IN_XLS, sheet_name=0)

    # Locate columns
    cols = {c.strip(): c for c in df.columns}
    cas_col = next((cols[k] for k in cols if k.lower() == "cas no."), None)
    result_col = next((cols[k] for k in cols if k.lower() in {"ames overall", "ames overall result"}), None)
    name_col = None
    for k in POSSIBLE_NAME_COLS:
        if k in cols:
            name_col = cols[k]
            break

    if cas_col is None or result_col is None:
        raise KeyError("Couldn't find 'CAS No.' and/or 'Ames Overall' columns in ECVAM_positive.xls.")

    # Prepare rows
    out_rows = []
    cache = {}  # cache by CAS first, then by Name

    for idx, row in df.iterrows():
        cas = extract_first_cas(row.get(cas_col))
        name = str(row.get(name_col)).strip() if (name_col and pd.notna(row.get(name_col))) else None
        ames = row.get(result_col)

        key = cas or (f"name::{name}" if name else None)
        smi, source = (None, "")
        if key in cache:
            smi, source = cache[key]
        else:
            smi, source = lookup_smiles(cas, name)
            cache[key] = (smi, source)

        out_rows.append({
            "CAS": cas,
            "Name": name,
            "AMES_Result": ames,
            "SMILES": smi,
            "SMILES_Source": source
        })

        # Light throttle
        time.sleep(SLEEP_BETWEEN_CALLS)

    out_df = pd.DataFrame(out_rows)
    # Optional: keep only rows where we actually found a CAS (drop completely empty)
    out_df = out_df.dropna(subset=["CAS"], how="all")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        out_df.to_excel(w, index=False, sheet_name="ECVAM_positive_with_SMILES")

    print(f"Wrote: {OUT_XLSX.resolve()}")
    print(f"Rows: {len(out_df)} | Found SMILES for: {(out_df['SMILES'].notna()).sum()}")

if __name__ == "__main__":
    main()
