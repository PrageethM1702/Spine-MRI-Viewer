import os
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARTICIPANTS_TSV = os.path.join(_BASE_DIR, "data", "data-multi-subject", "participants.tsv")
CACHE_DIR = os.path.join(_BASE_DIR, "data", "cache")


def load_participants() -> pd.DataFrame:
    if not os.path.exists(PARTICIPANTS_TSV):
        return pd.DataFrame(columns=["participant_id", "manufacturer"])
    return pd.read_csv(PARTICIPANTS_TSV, sep="\t")


def get_subjects_by_vendor(vendor: str = "All") -> list[str]:
    df = load_participants()
    if df.empty:
        return []
    df["manufacturer"] = df["manufacturer"].fillna("").astype(str)
    if vendor == "All":
        return sorted(df["participant_id"].tolist())
    if vendor == "Other":
        mask = ~df["manufacturer"].str.contains("GE|Philips|Siemens", case=False)
        return sorted(df[mask]["participant_id"].tolist())
    mask = df["manufacturer"].str.contains(vendor, case=False, na=False)
    return sorted(df[mask]["participant_id"].tolist())


def get_vendor_for_subject(subject: str) -> str:
    df = load_participants()
    row = df[df["participant_id"] == subject]
    if row.empty:
        return "Unknown"
    return str(row.iloc[0].get("manufacturer", "Unknown") or "Unknown")


def get_t2_path(subject: str) -> str | None:
    if not subject or subject == "No data":
        return None
    path = os.path.join(CACHE_DIR, subject, "anat", f"{subject}_T2w.nii.gz")
    return path if os.path.exists(path) and os.path.getsize(path) > 1024 else None