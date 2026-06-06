import os
import pandas as pd
from datetime import datetime

EXPORT_DIR = "exports"


def export_csv(data: dict, filename_prefix="measurements"):
    os.makedirs(EXPORT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"

    path = os.path.join(EXPORT_DIR, filename)

    df = pd.DataFrame([data])
    df.to_csv(path, index=False)

    return path


def export_metadata(subject_id, vendor, path):
    return export_csv({
        "subject": subject_id,
        "vendor": vendor,
        "nifti_path": path
    }, "subject_metadata")