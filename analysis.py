import os
import io
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_participants, get_t2_path

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_subject_json_path(subject: str) -> str:
    return os.path.join(
        _BASE_DIR, "data", "data-multi-subject",
        subject, "anat", f"{subject}_T2w.json"
    )


def get_field_strength(subject: str) -> str:
    try:
        import json
        p = _get_subject_json_path(subject)
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            fs = d.get("MagneticFieldStrength", "n/a")
            return f"{fs}T" if fs != "n/a" else "n/a"
    except Exception:
        pass
    return "n/a"


def downsample_nifti_if_needed(nifti_path: str, threshold: int = 256) -> str:
    img = nib.load(nifti_path)
    shape = img.shape[:3]
    if all(d <= threshold for d in shape):
        return nifti_path

    data = img.get_fdata()
    data_ds = data[::2, ::2, ::2]
    hdr = img.header.copy()
    zooms = np.array(img.header.get_zooms()[:3])
    hdr.set_zooms(tuple(zooms * 2) + tuple(img.header.get_zooms()[3:]))
    affine = img.affine.copy()
    affine[:3, :3] = affine[:3, :3] @ np.diag([2, 2, 2])
    new_img = nib.Nifti1Image(data_ds, affine, hdr)

    out_path = nifti_path.replace(".nii.gz", "_ds.nii.gz").replace(".nii", "_ds.nii")
    if not os.path.exists(out_path):
        nib.save(new_img, out_path)
    return out_path


def export_t2_middle_slice_png(nifti_path: str) -> bytes | None:
    if not nifti_path or not os.path.exists(nifti_path):
        return None
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        if data is None or len(data.shape) < 3:
            return None
        z = data.shape[2] // 2
        slice_img = data[:, :, z]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(slice_img.T, cmap="gray", origin="lower")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()
    except Exception as e:
        print("PNG export failed:", e)
        return None


def get_subject_metrics(subject: str, nifti_path: str, vendor: str) -> dict:
    try:
        img = nib.load(nifti_path)
        data = img.get_fdata()
        zooms = img.header.get_zooms()[:3]
        voxel_dims = f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f} mm"
        mean_intensity = float(np.mean(data))
    except Exception:
        voxel_dims = "n/a"
        mean_intensity = float("nan")

    return {
        "subject_id": subject,
        "vendor": vendor,
        "field_strength": get_field_strength(subject),
        "mean_intensity": round(mean_intensity, 4),
        "voxel_dims": voxel_dims,
    }


_MAX_SUBJECTS_PER_VENDOR = 8  # hard cap — prevents UI freeze on large cohorts


def compute_vendor_intensity_summary(vendor_filter: str = "All") -> pd.DataFrame:
    df = load_participants()
    if df is None or df.empty:
        return pd.DataFrame()

    results = []
    vendor_counts: dict[str, int] = {}

    for _, row in df.iterrows():
        subject = row.get("participant_id")
        vendor = str(row.get("manufacturer", "Unknown") or "Unknown")

        if vendor_filter != "All":
            if vendor_filter.lower() not in vendor.lower():
                continue

        # Per-vendor cap: never load more than _MAX_SUBJECTS_PER_VENDOR volumes
        # for any single vendor. This is the root cause of the Philips freeze —
        # Philips has 50 subjects and loading all of them blocks the reactive cell.
        vendor_key = vendor.lower()
        if vendor_counts.get(vendor_key, 0) >= _MAX_SUBJECTS_PER_VENDOR:
            continue

        path = get_t2_path(subject)
        if not path:
            continue

        try:
            img = nib.load(path)
            # Use a proxy stat: sample every 4th voxel along each axis so we
            # get a good mean estimate without loading the full array.
            data = img.dataobj[::4, ::4, ::4].astype(np.float32)
            results.append({
                "subject": subject,
                "vendor": vendor,
                "mean_intensity": float(np.mean(data)),
                "std_intensity": float(np.std(data)),
            })
            vendor_counts[vendor_key] = vendor_counts.get(vendor_key, 0) + 1
        except Exception:
            continue

    return pd.DataFrame(results)


def build_cohort_plotly_fig(vendor_filter: str = "All"):
    import plotly.graph_objects as go

    df = compute_vendor_intensity_summary(vendor_filter)
    if df is None or df.empty:
        return None

    fig = go.Figure()

    if vendor_filter == "All":
        for v in sorted(df["vendor"].dropna().unique()):
            sub = df[df["vendor"] == v]["mean_intensity"]
            fig.add_trace(go.Histogram(
                x=sub, name=v, opacity=0.7, nbinsx=20,
            ))
        fig.update_layout(
            barmode="overlay",
            title=f"Mean Signal Intensity by Vendor (up to {_MAX_SUBJECTS_PER_VENDOR} subjects/vendor)",
            xaxis_title="Mean Intensity",
            yaxis_title="Count",
            legend_title="Vendor",
        )
    else:
        sub = df["mean_intensity"]
        # Always use a plain histogram — create_distplot blocks the UI thread
        # and requires scipy; a simple histogram is faster and equally informative.
        fig.add_trace(go.Histogram(
            x=sub,
            name=vendor_filter,
            nbinsx=max(5, len(sub)),
            opacity=0.8,
            marker_color="#4C78A8",
        ))
        fig.update_layout(
            title=f"Signal Intensity Distribution — {vendor_filter} ({len(sub)} subjects sampled)",
            xaxis_title="Mean Intensity",
            yaxis_title="Count",
        )

    fig.update_layout(height=400, margin=dict(l=40, r=20, t=50, b=40))
    return fig
