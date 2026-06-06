import os
import re
from glob import glob

import pandas as pd


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_METRICS_UPLOAD_DIR = os.path.join(_DATA_DIR, "metrics")


_SUBJECT_COLUMNS = ("participant_id", "subject", "subject_id", "participant", "sub")
_LEVEL_COLUMNS = ("vertebral_level", "level", "label", "vert_level", "disc_level")


def _norm(value) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def _read_table(path: str) -> pd.DataFrame | None:
    try:
        sep = "\t" if path.lower().endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    except Exception:
        return None


def metric_table_paths() -> list[str]:
    patterns = (
        os.path.join(_DATA_DIR, "**", "*.csv"),
        os.path.join(_DATA_DIR, "**", "*.tsv"),
    )
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob(pattern, recursive=True))
    ignored = {"participants.csv", "participants.tsv"}
    return sorted(
        p for p in set(paths)
        if os.path.basename(p).lower() not in ignored
    )


def available_metric_tables() -> list[dict]:
    tables = []
    for path in metric_table_paths():
        df = _read_table(path)
        if df is None or df.empty:
            continue
        tables.append({
            "path": path,
            "name": os.path.relpath(path, _DATA_DIR),
            "rows": len(df),
            "columns": list(df.columns),
        })
    return tables


def save_metric_upload(name: str, contents: bytes) -> str | None:
    if not name.lower().endswith((".csv", ".tsv")):
        return None
    os.makedirs(_METRICS_UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(name).replace(" ", "_")
    path = os.path.join(_METRICS_UPLOAD_DIR, safe_name)
    with open(path, "wb") as f:
        f.write(contents)
    return path


def _find_column(columns, candidates) -> str | None:
    by_norm = {_norm(c): c for c in columns}
    for candidate in candidates:
        key = _norm(candidate)
        if key in by_norm:
            return by_norm[key]
    for col in columns:
        c = _norm(col)
        if any(_norm(candidate) in c for candidate in candidates):
            return col
    return None


def metrics_for_subject_level(subject: str, level: str | None = None) -> tuple[pd.DataFrame, str]:
    frames = []
    for table in available_metric_tables():
        df = _read_table(table["path"])
        if df is None or df.empty:
            continue

        subject_col = _find_column(df.columns, _SUBJECT_COLUMNS)
        if subject_col:
            df = df[df[subject_col].astype(str).map(_norm) == _norm(subject)]
        else:
            name_has_subject = _norm(subject) in _norm(os.path.basename(table["path"]))
            if not name_has_subject:
                continue

        level_col = _find_column(df.columns, _LEVEL_COLUMNS)
        if level and level_col:
            wanted = _norm(level)
            df = df[df[level_col].astype(str).map(_norm).str.contains(re.escape(wanted), na=False)]

        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "metric_source", table["name"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(), "No metric CSV/TSV rows found for this subject/level."
    return pd.concat(frames, ignore_index=True, sort=False), "ok"
