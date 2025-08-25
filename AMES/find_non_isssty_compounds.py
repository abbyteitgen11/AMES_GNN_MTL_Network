import re
import pandas as pd

# --- CAS utilities ---
CAS_REGEX = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

def extract_cas(cell):
    """Return all CAS-like tokens from a cell (handles strings, numbers, NaN)."""
    if pd.isna(cell):
        return []
    s = str(cell)
    s = s.replace(" 00:00:00", "")  # strip midnight timestamps from Excel auto-date
    return CAS_REGEX.findall(s)

def cas_checksum_is_valid(cas: str) -> bool:
    """
    Validate CAS checksum:
    - Remove hyphens, last digit is the check digit.
    - Multiply preceding digits from right-to-left by 1,2,3,...
    - Sum products; sum % 10 must equal check digit.
    """
    digits = cas.replace("-", "")
    if not digits.isdigit() or len(digits) < 3:
        return False
    check = int(digits[-1])
    body = digits[:-1][::-1]  # reverse for weighting 1..n
    total = sum((i + 1) * int(d) for i, d in enumerate(body))
    return total % 10 == check

def cell_has_only_that_token(cell_str: str, token: str) -> bool:
    """True iff the string equals exactly the token (no extra text)."""
    return cell_str.strip().replace(" 00:00:00", "") == token

def normalize_cas_set(series: pd.Series) -> set:
    """
    Collect all CAS tokens in a column (every token in multi-token cells is counted).
    """
    out = set()
    for cell in series.dropna().tolist():
        for cas in extract_cas(cell):
            out.add(cas.strip())
    return out

def main():
    # --- read input ---
    df = pd.read_excel('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/dataset_comparison.xlsx')

    # --- data-quality pass: per-token validation for each column ---
    quality_rows = []  # Column, Row, OriginalValue, Token, IsValidChecksum, OnlyTokenInCell, Notes

    for col in df.columns:
        col_series = df[col]
        for idx, val in col_series.items():
            if pd.isna(val):
                quality_rows.append({
                    "Column": col,
                    "Row": idx,
                    "OriginalValue": None,
                    "Token": None,
                    "IsValidChecksum": None,
                    "OnlyTokenInCell": None,
                    "Notes": "NaN"
                })
                continue

            s = str(val).strip().replace(" 00:00:00", "")
            tokens = extract_cas(s)

            if len(tokens) == 0:
                quality_rows.append({
                    "Column": col,
                    "Row": idx,
                    "OriginalValue": s,
                    "Token": None,
                    "IsValidChecksum": None,
                    "OnlyTokenInCell": None,
                    "Notes": "No CAS token found"
                })
                continue

            # For cells with one or multiple tokens: validate EACH token
            for t in tokens:
                quality_rows.append({
                    "Column": col,
                    "Row": idx,
                    "OriginalValue": s,
                    "Token": t,
                    "IsValidChecksum": cas_checksum_is_valid(t),
                    "OnlyTokenInCell": cell_has_only_that_token(s, t),
                    "Notes": ("Extra text around CAS" if not cell_has_only_that_token(s, t) else "OK")
                })

    quality_df = pd.DataFrame(
        quality_rows,
        columns=["Column", "Row", "OriginalValue", "Token", "IsValidChecksum", "OnlyTokenInCell", "Notes"]
    )

    # --- original overlap logic (unchanged) ---
    isssty_col = df.columns[0]
    isssty_set = normalize_cas_set(df[isssty_col])

    unique_dict = {}
    total_counts = {}
    for col in df.columns[1:]:
        other_set = normalize_cas_set(df[col])
        uniques = sorted(other_set - isssty_set)
        unique_dict[col] = uniques
        total_counts[col] = len(other_set)

    # build summary
    summary_rows = []
    for col, uniques in unique_dict.items():
        summary_rows.append({
            "Dataset": col,
            "Total in dataset (distinct CAS)": total_counts[col],
            "Not in ISSSTY (distinct CAS)": len(uniques)
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Not in ISSSTY (distinct CAS)", ascending=False)

    # align columns to same length for a single sheet view
    max_len = max((len(v) for v in unique_dict.values()), default=0)
    unique_df = pd.DataFrame({k: (v + [None]*(max_len - len(v))) for k, v in unique_dict.items()})

    # write output with an extra Data_quality sheet
    with pd.ExcelWriter('/Users/abigailteitgen/Dropbox/Postdoc/AMES_GNN_MTL_Network/Additional_data/unique.xlsx', engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        unique_df.to_excel(writer, sheet_name="Not_in_ISSSTY", index=False)
        quality_df.to_excel(writer, sheet_name="Data_quality", index=False)

if __name__ == "__main__":
    main()

