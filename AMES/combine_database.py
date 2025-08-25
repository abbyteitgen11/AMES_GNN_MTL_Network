#!/usr/bin/env python3
"""
Combine CAS from 'Not_in_ISSSTY' with ECVAM+/ECVAM-/ToxBenchmark records.

Inputs (same folder as this script by default):
- unique.xlsx                  (sheet: "Not_in_ISSSTY")
- ECVAM_positive.xls           (cols: 'CAS No.' , 'Ames Overall', optional 'SMILES')
- ECVAM_negative.xlsx          (cols: 'CAS No.,' , 'AMES Overall', optional 'SMILES')
- Tox_benchmark.xls            (cols: 'CAS_NO', 'Activity')
- Tox_benchmark_smiles.smi     (format: "SMILES<space-or-tab>identifier", identifier contains CAS)

Output:
- non_isssty_combined.xlsx     (sheet: "Combined", cols: CAS, SMILES, AMES_Result, Source)
"""

import re
from pathlib import Path
import pandas as pd

# ---------- Config (edit paths if your files live elsewhere) ----------
UNIQUE_XLSX = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/unique.xlsx")
ECVAM_POS_XLS = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/ECVAM_positive_simple_with_smiles_cleaned.xlsx")
ECVAM_NEG_XLSX = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/ECVAM_negative_simple.xlsx")
TOXBENCH_XLS = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/Tox_benchmark.xls")
TOXBENCH_SMI = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/Tox_benchmark_smiles.smi")
OUT_XLSX = Path("/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/non_isssty_combined.xlsx")

CAS_REGEX = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

# ---------- Helpers ----------
def extract_cas_all(cell) -> list[str]:
    """Return all CAS-like tokens from a cell."""
    if pd.isna(cell):
        return []
    s = str(cell).replace(" 00:00:00", "")
    return CAS_REGEX.findall(s)

def normalize_cas_unique(series: pd.Series) -> set[str]:
    """Collect distinct CAS tokens from a column, including multi-token cells."""
    out = set()
    for cell in series.dropna():
        out.update(extract_cas_all(cell))
    return out

def read_not_in_isssty(unique_path: Path) -> dict[str, set[str]]:
    """Read unique.xlsx / Not_in_ISSSTY and collect target CAS per dataset column."""
    df = pd.read_excel(unique_path, sheet_name="Not_in_ISSSTY")
    cas_sets = {}
    for col in df.columns:
        norm = col.strip().lower()
        if norm in {"ecvam positive", "ecvam negative", "toxbenchmark"}:
            cas_sets[norm] = normalize_cas_unique(df[col])
    return cas_sets

def load_ecvam_positive(path: Path) -> pd.DataFrame:
    """Load ECVAM positive: CAS, SMILES (if present), Ames Overall."""
    # Requires xlrd for .xls
    df = pd.read_excel(path, sheet_name=0)
    cols = {c.strip(): c for c in df.columns}
    cas_col = next((cols[k] for k in cols if k.lower() == "cas no."), None)
    result_col = next((cols[k] for k in cols if k.lower() in {"ames overall", "ames overall result"}), None)
    smiles_col = next((cols[k] for k in cols if k.lower() == "smiles"), None)
    if cas_col is None or result_col is None:
        raise KeyError("ECVAM_positive: couldn't find 'CAS No.' and/or 'Ames Overall' columns.")
    out = pd.DataFrame({
        "CAS": df[cas_col].apply(lambda x: extract_cas_all(x)[0] if extract_cas_all(x) else pd.NA),
        "SMILES": df[smiles_col] if smiles_col is not None else pd.NA,
        "AMES_Result": df[result_col],
        "Source": "ECVAM positive"
    })
    return out.dropna(subset=["CAS"])

def load_ecvam_negative(path: Path) -> pd.DataFrame:
    """Load ECVAM negative: CAS, SMILES (if present), AMES Overall."""
    df = pd.read_excel(path, sheet_name=0)
    cols = {c.strip(): c for c in df.columns}
    # 'CAS No.,' includes a trailing comma; normalize by removing commas
    cas_col = next((cols[k] for k in cols if k.lower().replace(",", "") == "cas no."), None)
    # The user note says 'AMES Overall' (upper AMES). Be flexible and accept case variants.
    result_col = next((cols[k] for k in cols if k.strip() in {"AMES Overall", "Ames Overall"} or k.strip().lower() == "ames overall"), None)
    smiles_col = next((cols[k] for k in cols if k.lower() == "smiles"), None)
    if cas_col is None or result_col is None:
        raise KeyError("ECVAM_negative: couldn't find 'CAS No.,' and/or 'AMES Overall' columns.")
    out = pd.DataFrame({
        "CAS": df[cas_col].apply(lambda x: extract_cas_all(x)[0] if extract_cas_all(x) else pd.NA),
        "SMILES": df[smiles_col] if smiles_col is not None else pd.NA,
        "AMES_Result": df[result_col],
        "Source": "ECVAM negative"
    })
    return out.dropna(subset=["CAS"])

def read_smi_as_map(path: Path) -> dict[str, str]:
    """
    Parse .smi assumed as 'SMILES<space|tab>identifier'. We try to extract a CAS from the identifier.
    If the identifier has a CAS, use it as the key; else use the raw identifier.
    """
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line, maxsplit=1)  # split on first whitespace run
            smiles, ident = (parts[0], parts[1]) if len(parts) == 2 else (parts[0], "")
            cas_tokens = CAS_REGEX.findall(ident)
            key = cas_tokens[0] if cas_tokens else ident.strip()
            if key:
                mapping[key] = smiles
    return mapping

def load_toxbenchmark(xls_path: Path, smi_path: Path) -> pd.DataFrame:
    """Load ToxBenchmark: CAS_NO, Activity + SMILES from .smi file."""
    df = pd.read_excel(xls_path, sheet_name=0)
    cols = {c.strip(): c for c in df.columns}
    cas_col = next((cols[k] for k in cols if k.strip().lower() == "cas_no"), None)
    act_col = next((cols[k] for k in cols if k.strip().lower() == "activity"), None)
    if cas_col is None or act_col is None:
        raise KeyError("ToxBenchmark: couldn't find 'CAS_NO' and/or 'Activity' columns.")
    smiles_map = read_smi_as_map(smi_path)

    cas_extracted = df[cas_col].apply(lambda x: extract_cas_all(x)[0] if extract_cas_all(x) else pd.NA)
    smiles_series = cas_extracted.map(lambda k: smiles_map.get(str(k), pd.NA))

    out = pd.DataFrame({
        "CAS": cas_extracted,
        "SMILES": smiles_series,
        "AMES_Result": df[act_col],
        "Source": "ToxBenchmark"
    })
    return out.dropna(subset=["CAS"])

def clean_label(x):
    if pd.isna(x):
        return x
    return str(x).strip()

def main():
    # Load the “not in ISSSTY” sets per dataset
    cas_sets = read_not_in_isssty(UNIQUE_XLSX)
    # Load datasets
    ecvam_pos_df = load_ecvam_positive(ECVAM_POS_XLS)
    ecvam_neg_df = load_ecvam_negative(ECVAM_NEG_XLSX)
    tox_df = load_toxbenchmark(TOXBENCH_XLS, TOXBENCH_SMI)

    # Filter by the specific CAS lists from Not_in_ISSSTY
    pos_target = cas_sets.get("ecvam positive", set())
    neg_target = cas_sets.get("ecvam negative", set())
    tox_target = cas_sets.get("toxbenchmark", set())

    ecvam_pos_not = ecvam_pos_df[ecvam_pos_df["CAS"].isin(pos_target)].copy()
    ecvam_neg_not = ecvam_neg_df[ecvam_neg_df["CAS"].isin(neg_target)].copy()
    tox_not = tox_df[tox_df["CAS"].isin(tox_target)].copy()

    combined = pd.concat([ecvam_pos_not, ecvam_neg_not, tox_not], ignore_index=True)
    combined["AMES_Result"] = combined["AMES_Result"].map(clean_label)
    combined = combined.drop_duplicates(subset=["CAS", "Source"], keep="first")

    # Save
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
        combined.to_excel(writer, index=False, sheet_name="Combined")

    print(f"Wrote: {OUT_XLSX.resolve()}")
    print(f"Rows: {len(combined)}")

if __name__ == "__main__":
    main()
