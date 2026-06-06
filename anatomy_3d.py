import os

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import binary_erosion

from overlays import VERTEBRAL_LEVELS


def _load_mask_points(path: str, max_points: int = 12000) -> tuple[np.ndarray, np.ndarray] | None:
    if not path or not os.path.exists(path):
        return None
    img = nib.load(path)
    data = np.asanyarray(img.dataobj) > 0
    if not np.any(data):
        return None

    surface = data & ~binary_erosion(data, iterations=1)
    pts = np.argwhere(surface if np.any(surface) else data)
    if len(pts) > max_points:
        step = max(1, len(pts) // max_points)
        pts = pts[::step]
    world = nib.affines.apply_affine(img.affine, pts)
    return world, pts


def _label_centroids(path: str, max_labels: int = 30) -> list[tuple[str, np.ndarray]]:
    if not path or not os.path.exists(path):
        return []
    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    values = [
        int(v) for v in np.unique(data)
        if v > 0 and np.isfinite(v)
    ][:max_labels]
    labels = []
    for value in values:
        coords = np.argwhere(data == value)
        if coords.size == 0:
            continue
        centroid_vox = coords.mean(axis=0)
        centroid_mm = nib.affines.apply_affine(img.affine, centroid_vox)
        name = VERTEBRAL_LEVELS[value - 1] if 1 <= value <= len(VERTEBRAL_LEVELS) else f"L{value}"
        labels.append((name, centroid_mm))
    return labels


def build_anatomy_3d_figure(
    anatomy_path: str,
    seg_path: str | None = None,
    labels_path: str | None = None,
    pam50_path: str | None = None,
):
    """Build a lightweight 3D scene from native-space overlay geometry."""
    fig = go.Figure()

    seg = _load_mask_points(seg_path, max_points=16000)
    if seg is not None:
        xyz, _ = seg
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            name="Spinal cord boundary",
            marker=dict(size=2.2, color="#ef4444", opacity=0.42),
        ))

    pam50 = _load_mask_points(pam50_path, max_points=14000)
    if pam50 is not None:
        xyz, _ = pam50
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode="markers",
            name="PAM50 cord template",
            marker=dict(size=2.0, color="#2563eb", opacity=0.42),
        ))

    label_items = _label_centroids(labels_path)
    if label_items:
        names = [item[0] for item in label_items]
        pts = np.vstack([item[1] for item in label_items])
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers+text",
            name="Vertebral labels",
            text=names,
            customdata=names,
            textposition="middle right",
            marker=dict(size=5, color="#f59e0b", opacity=0.95),
            textfont=dict(color="#f59e0b", size=12),
        ))

    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(title="X mm", showbackground=False),
            yaxis=dict(title="Y mm", showbackground=False),
            zaxis=dict(title="Z mm", showbackground=False),
            aspectmode="data",
            bgcolor="black",
        ),
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="#e5e7eb"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            font=dict(color="#e5e7eb"),
        ),
    )
    return fig


def export_3d_figure_png(fig) -> bytes | None:
    try:
        return fig.to_image(format="png", width=1400, height=900, scale=2)
    except Exception:
        return None
