import os
import glob
import subprocess
import urllib.request
import hashlib
import shutil
from fnmatch import fnmatch

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, binary_dilation, binary_erosion, gaussian_filter

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_BASE_DIR, "data", "cache")
_REPO_DIR = os.path.join(_BASE_DIR, "data", "data-multi-subject")

# spine-generic derivative locations.
_DERIV_LABELS = os.path.join(_REPO_DIR, "derivatives", "labels")
_DERIV_SOFTSEG_BIN = os.path.join(_REPO_DIR, "derivatives", "labels_softseg_bin")
_DERIV_SOFTSEG = os.path.join(_REPO_DIR, "derivatives", "labels_softseg")
_PAM50_DIR = os.path.join(_BASE_DIR, "data", "pam50")
_ALIGNED_DIR = os.path.join(_CACHE_DIR, "aligned_overlays")
_PAM50_NATIVE_DIR = os.path.join(_CACHE_DIR, "pam50_native")
_ALIGN_VERSION = "v4_direct_multivolume"

OVERLAY_OPACITY = 0.4
_PAM50_TEMPLATE_NAME = "PAM50_t2.nii.gz"
_PAM50_CORD_NAME = "PAM50_cord.nii.gz"
_PAM50_LEVELS_NAME = "PAM50_levels.nii.gz"
_PAM50_DOWNLOADS = (_PAM50_TEMPLATE_NAME, _PAM50_CORD_NAME, _PAM50_LEVELS_NAME)
_PAM50_GITHUB_RAW = (
    "https://raw.githubusercontent.com/spinalcordtoolbox/PAM50/"
    "r20231011/template/"
)

VERTEBRAL_LEVELS = [
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5",
]

_SEG_PATTERNS = [
    (_DERIV_SOFTSEG_BIN, "{sub}_T2w_desc-softseg_label-SC_seg.nii.gz"),
    (_DERIV_SOFTSEG_BIN, "{sub}_*desc-softseg*label-SC*_seg.nii.gz"),
    (_DERIV_SOFTSEG_BIN, "{sub}_*label-SC*_seg.nii.gz"),
    (_DERIV_SOFTSEG_BIN, "{sub}_*_seg.nii.gz"),
    (_DERIV_SOFTSEG, "{sub}_*desc-softseg*label-SC*_seg.nii.gz"),
    (_DERIV_SOFTSEG, "{sub}_*label-SC*_seg.nii.gz"),
    (_DERIV_LABELS, "{sub}_T2w_label-SC_seg.nii.gz"),
    (_DERIV_LABELS, "{sub}_*label-SC*_seg.nii.gz"),
    (_DERIV_LABELS, "{sub}_*_seg-manual.nii.gz"),
    (_DERIV_LABELS, "{sub}_*_seg.nii.gz"),
    (_CACHE_DIR, "{sub}_*label-SC*_seg.nii.gz"),
    (_CACHE_DIR, "{sub}_*_seg-manual.nii.gz"),
    (_CACHE_DIR, "{sub}_*_seg.nii.gz"),
]

_LABELS_PATTERNS = [
    (_DERIV_LABELS, "{sub}_T2w_label-discs_dlabel.nii.gz"),
    (_DERIV_LABELS, "{sub}_*label-discs_dlabel.nii.gz"),
    (_DERIV_LABELS, "{sub}_*label-disc*_dlabel.nii.gz"),
    (_DERIV_LABELS, "{sub}_*label-vert*_dlabel.nii.gz"),
    (_DERIV_LABELS, "{sub}_*label-vertebrae*_dlabel.nii.gz"),
    (_DERIV_LABELS, "{sub}_*label-spine_dseg.nii.gz"),
    (_DERIV_LABELS, "{sub}_*_labels.nii.gz"),
    (_DERIV_LABELS, "{sub}_*_dlabel.nii.gz"),
    (_CACHE_DIR, "{sub}_*label-discs*.nii.gz"),
    (_CACHE_DIR, "{sub}_*label-vert*.nii.gz"),
    (_CACHE_DIR, "{sub}_*labels.nii.gz"),
    (_CACHE_DIR, "{sub}_*_dlabel.nii.gz"),
]

_SEG_TREE_PATTERNS = (
    "derivatives/labels_softseg_bin/{sub}/anat/{sub}_T2w_desc-softseg_label-SC_seg.nii.gz",
    "derivatives/labels_softseg_bin/{sub}/anat/{sub}_*label-SC*_seg.nii.gz",
    "derivatives/labels_softseg_bin/{sub}/anat/{sub}_*_seg.nii.gz",
    "derivatives/labels_softseg/{sub}/anat/{sub}_*label-SC*_seg.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_T2w*_seg-manual.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*T2w*_seg-manual.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*label-SC*_seg.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*_seg.nii.gz",
)

_LABELS_TREE_PATTERNS = (
    "derivatives/labels/{sub}/anat/{sub}_T2w*_labels-manual.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*T2w*_labels-manual.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*T2w*label-manual.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*label-discs_dlabel.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*label-disc*_dlabel.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*label-vert*_dlabel.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*_labels.nii.gz",
    "derivatives/labels/{sub}/anat/{sub}_*_dlabel.nii.gz",
)


def _bar_update(bar, increment: int, subtitle: str) -> None:
    if bar is not None:
        bar.update(increment=increment, subtitle=subtitle)


def _existing_nifti(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1024


def _safe_key(*parts: str) -> str:
    raw = "|".join(os.path.abspath(p) for p in parts if p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _reference_contrast(reference_path: str | None) -> str | None:
    if not reference_path:
        return None
    name = os.path.basename(reference_path)
    for contrast in ("T2w", "T1w", "T2star", "MTS"):
        if f"_{contrast}" in name or f"-{contrast}" in name:
            return contrast
    return None


def _same_contrast(path: str, reference_path: str | None) -> bool:
    contrast = _reference_contrast(reference_path)
    if not contrast:
        return True
    name = os.path.basename(path)
    return f"_{contrast}" in name or f"-{contrast}" in name


def _rank_candidate(path: str, reference_path: str | None) -> tuple[int, int, int, str]:
    name = os.path.basename(path)
    same_contrast = 0 if _same_contrast(path, reference_path) else 1
    manual = 0 if "manual" in name else 1
    t2w = 0 if "_T2w" in name else 1
    return (same_contrast, manual, t2w, name)


def _nifti_compatible(path: str, reference_path: str) -> bool:
    try:
        src = nib.load(path)
        ref = nib.load(reference_path)
        return src.shape[:3] == ref.shape[:3] and np.allclose(src.affine, ref.affine, atol=1e-3)
    except Exception:
        return False


def _world_bbox_from_voxels(img: nib.spatialimages.SpatialImage, data: np.ndarray | None = None):
    shape = img.shape[:3]
    if data is not None and np.any(data):
        coords = np.argwhere(data)
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
    else:
        mins = np.array([0, 0, 0])
        maxs = np.array(shape) - 1
    corners = np.array([
        [mins[0], mins[1], mins[2]],
        [mins[0], mins[1], maxs[2]],
        [mins[0], maxs[1], mins[2]],
        [mins[0], maxs[1], maxs[2]],
        [maxs[0], mins[1], mins[2]],
        [maxs[0], mins[1], maxs[2]],
        [maxs[0], maxs[1], mins[2]],
        [maxs[0], maxs[1], maxs[2]],
    ])
    world = nib.affines.apply_affine(img.affine, corners)
    return world.min(axis=0), world.max(axis=0)


def _has_world_overlap(path: str, reference_path: str) -> bool:
    try:
        src = nib.load(path)
        ref = nib.load(reference_path)
        src_data = np.asanyarray(src.dataobj) > 0
        if not np.any(src_data):
            return False
        src_min, src_max = _world_bbox_from_voxels(src, src_data)
        ref_min, ref_max = _world_bbox_from_voxels(ref)
        overlap_min = np.maximum(src_min, ref_min)
        overlap_max = np.minimum(src_max, ref_max)
        return bool(np.all(overlap_max > overlap_min))
    except Exception:
        return False


def _resample_to_reference(
    source_path: str,
    reference_path: str,
    out_path: str,
    *,
    order: int,
    threshold: float | None = None,
) -> str | None:
    """
    Resample source NIfTI into the voxel grid of reference NIfTI.

    ONLY use this when source and reference share the same world coordinate
    system (e.g. both are subject-space images: seg vs T2w, labels vs T2w).

    DO NOT use this for PAM50 → subject-space resampling.  PAM50 lives in
    MNI/ICBM152 space (straightened template), which has completely different
    world coordinates from any subject scanner space.  Without SCT's non-linear
    warp, affine resampling of PAM50 into subject space is meaningless and
    produces the large misaligned blob seen in the viewer.
    """
    try:
        src = nib.as_closest_canonical(nib.load(source_path))
        ref = nib.as_closest_canonical(nib.load(reference_path))
        data = np.asanyarray(src.dataobj)

        # M maps reference voxel coords → source voxel coords (4×4 homogeneous).
        # scipy.ndimage.affine_transform(data, A, b): output[o] = data[A@o + b]
        # linalg.solve is numerically more stable than inv() @
        M = np.linalg.solve(src.affine, ref.affine)

        sampled = affine_transform(
            data,
            matrix=M[:3, :3],
            offset=M[:3, 3],
            output_shape=ref.shape[:3],
            order=order,
            mode="constant",
            cval=0,
        )
        if threshold is not None:
            sampled = (sampled >= threshold).astype(np.uint8)
        elif order == 0:
            sampled = np.rint(sampled).astype(np.int16)
        else:
            sampled = sampled.astype(np.float32)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        header = ref.header.copy()
        header.set_data_dtype(sampled.dtype)
        nib.save(nib.Nifti1Image(sampled, ref.affine, header), out_path)
        return out_path if _existing_nifti(out_path) else None
    except Exception:
        return None


def _aligned_overlay_path(source_path: str, reference_path: str, kind: str) -> str | None:
    """
    Align a subject-space overlay (seg or labels) into the reference voxel grid.

    Both source and reference must be in the same world coordinate system
    (subject scanner space).  This is valid for seg→T2w and labels→T2w.
    """
    if not source_path or not reference_path:
        return None
    if _nifti_compatible(source_path, reference_path):
        return source_path

    ext = ".nii.gz" if source_path.endswith(".nii.gz") else ".nii"
    out = os.path.join(
        _ALIGNED_DIR,
        f"{os.path.basename(source_path).replace('.nii.gz', '').replace('.nii', '')}"
        f"_to_{_safe_key(_ALIGN_VERSION, reference_path, source_path)}{ext}",
    )
    if _existing_nifti(out) and _nifti_compatible(out, reference_path):
        return out

    if kind == "seg":
        return _resample_to_reference(source_path, reference_path, out, order=0, threshold=0.5)
    if kind == "labels":
        return _resample_to_reference(source_path, reference_path, out, order=0)
    return _resample_to_reference(source_path, reference_path, out, order=1)


def _download_pam50_file(name: str) -> tuple[bool, str]:
    os.makedirs(_PAM50_DIR, exist_ok=True)
    dest = os.path.join(_PAM50_DIR, name)
    if _existing_nifti(dest):
        return True, dest
    try:
        with urllib.request.urlopen(_PAM50_GITHUB_RAW + name, timeout=120) as req:
            with open(dest, "wb") as f:
                while True:
                    buf = req.read(1024 * 64)
                    if not buf:
                        break
                    f.write(buf)
        return (_existing_nifti(dest), dest)
    except Exception as exc:
        if os.path.exists(dest):
            try:
                os.remove(dest)
            except OSError:
                pass
        return False, str(exc)


def _find_file(
    subject: str,
    patterns: list[tuple[str, str]],
    reference_path: str | None = None,
    *,
    require_same_contrast: bool = True,
) -> str | None:
    """Search for the first existing, non-empty file matching any pattern."""
    if not subject:
        return None

    for base_dir, fname_tpl in patterns:
        fname = fname_tpl.format(sub=subject)
        search_patterns = (
            os.path.join(base_dir, subject, "anat", fname),
            os.path.join(base_dir, subject, "**", fname),
            os.path.join(base_dir, "**", subject, "anat", fname),
            os.path.join(base_dir, "**", fname),
        )
        hits: list[str] = []
        for pat in search_patterns:
            hits.extend(
                p for p in glob.glob(pat, recursive=True)
                if _existing_nifti(p)
            )
        if require_same_contrast and reference_path:
            hits = [p for p in hits if _same_contrast(p, reference_path)]
        if hits:
            return sorted(set(hits), key=lambda p: _rank_candidate(p, reference_path))[0]

    return None


def get_seg_path(subject: str, reference_path: str | None = None) -> str | None:
    return _find_file(subject, _SEG_PATTERNS, reference_path)


def get_labels_path(subject: str, reference_path: str | None = None) -> str | None:
    return _find_file(subject, _LABELS_PATTERNS, reference_path)


def get_pam50_path() -> str | None:
    candidates = (
        os.path.join(_PAM50_DIR, "template", _PAM50_TEMPLATE_NAME),
        os.path.join(_PAM50_DIR, _PAM50_TEMPLATE_NAME),
    )
    for path in candidates:
        if _existing_nifti(path):
            return path
    return None


def get_pam50_cord_path() -> str | None:
    for path in (
        os.path.join(_PAM50_DIR, "template", _PAM50_CORD_NAME),
        os.path.join(_PAM50_DIR, _PAM50_CORD_NAME),
    ):
        if _existing_nifti(path):
            return path
    return None


def _sct_native_pam50(subject: str, reference_path: str) -> str | None:
    """
    Use SCT's sct_register_to_template + sct_warp_template to bring PAM50
    into subject space with the correct non-linear warp.  This is the only
    fully correct method — affine resampling of PAM50 is wrong because PAM50
    is a straightened MNI-space template, not a curved scanner-space image.
    """
    sct_register = shutil.which("sct_register_to_template")
    sct_warp = shutil.which("sct_warp_template")
    if not sct_register or not sct_warp:
        return None

    seg = _aligned_overlay_path(get_seg_path(subject, reference_path), reference_path, "seg")
    labels = _aligned_overlay_path(get_labels_path(subject, reference_path), reference_path, "labels")
    if not seg:
        return None

    out_dir = os.path.join(_PAM50_NATIVE_DIR, subject, _safe_key(reference_path))
    template_out = os.path.join(out_dir, "label", "template", _PAM50_TEMPLATE_NAME)
    if _existing_nifti(template_out) and _nifti_compatible(template_out, reference_path):
        return template_out

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sct_register,
        "-i", reference_path,
        "-s", seg,
        "-ref", "subject",
        "-ofolder", out_dir,
        "-c", "t2",
        "-r", "0",
        "-v", "0",
    ]
    if labels:
        cmd.extend(["-ldisc", labels])

    try:
        reg = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if reg.returncode != 0:
            return None
        warp = os.path.join(out_dir, "warp_template2anat.nii.gz")
        if not _existing_nifti(warp):
            return None
        warped = subprocess.run(
            [sct_warp, "-d", reference_path, "-w", warp,
             "-ofolder", os.path.join(out_dir, "label"), "-a", "0", "-v", "0"],
            capture_output=True, text=True, timeout=1200,
        )
        if warped.returncode == 0 and _existing_nifti(template_out):
            return template_out
    except Exception:
        return None
    return None


def _seg_based_cord_overlay(subject: str, reference_path: str) -> str | None:
    """
    Build a PAM50-style cord overlay directly from the subject's spinal cord
    segmentation — no PAM50 files needed and no MNI-space resampling.

    The PAM50_cord.nii.gz in MNI/template space cannot be resampled into
    subject space using an affine transform because:
      - PAM50 is a STRAIGHTENED template in ICBM152/MNI coordinate space.
      - Subject T2w images are CURVED and live in scanner (world) coordinate
        space with completely different origin, orientation, and scale.
      - scipy.ndimage.affine_transform maps voxel indices via a linear
        transform derived from the two affines, which only works correctly
        when both images share the same world coordinate space.
      - Applying it across PAM50→subject produces the large misaligned purple
        blob seen in the viewer (the cord appears far outside the spine).

    The only correct PAM50-to-subject mapping requires SCT's non-linear warp
    (sct_register_to_template + sct_warp_template).  When SCT is not
    available we instead build a cord-surface overlay purely from the
    subject's own segmentation, which is always in the correct anatomical
    location:

      1. Align the subject seg into the reference grid (affine — valid because
         both are in subject scanner space).
      2. Erode the seg 1 voxel inward to get a tight cord interior.
      3. Subtract to isolate the cord surface shell (1-2 voxel thick).
      4. Save as a binary mask with the reference affine → correctly aligned.
    """
    seg_src = get_seg_path(subject, reference_path)
    if not seg_src:
        return None

    out = os.path.join(
        _PAM50_NATIVE_DIR, subject,
        f"cord_surface_{_safe_key(reference_path, seg_src)}.nii.gz",
    )
    if _existing_nifti(out) and _nifti_compatible(out, reference_path):
        return out

    # Align seg into reference space (both are subject-space: valid affine resample)
    seg_ref = _aligned_overlay_path(seg_src, reference_path, "seg")
    if not seg_ref:
        return None

    try:
        ref = nib.as_closest_canonical(nib.load(reference_path))
        seg_img = nib.as_closest_canonical(nib.load(seg_ref))
        seg = np.asanyarray(seg_img.dataobj) > 0

        if not np.any(seg):
            return None

        # Get voxel sizes to determine structuring element for erosion
        zooms = np.abs(np.diag(ref.affine)[:3])  # mm per voxel

        # Erode by ~1 mm in each direction — creates the inner cord boundary
        # Using a 3×3×3 kernel, scaled so we erode roughly 1 voxel regardless
        # of resolution.
        eroded = binary_erosion(seg, iterations=1)

        # Surface shell = original seg minus eroded interior
        surface = seg & ~eroded

        # Also include a thin dilation outside the cord boundary for visibility
        dilated = binary_dilation(seg, iterations=1)
        outer_rim = dilated & ~seg

        # Combined: inner surface + outer rim = cord boundary shell
        cord_surface = (surface | outer_rim).astype(np.uint8)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        header = ref.header.copy()
        header.set_data_dtype(cord_surface.dtype)
        nib.save(nib.Nifti1Image(cord_surface, ref.affine, header), out)
        return out if _existing_nifti(out) else None
    except Exception:
        return None


def get_native_pam50_path(subject: str, reference_path: str) -> str | None:
    """
    Return a PAM50-style cord mask in subject/reference space.

    Priority:
      1. SCT-based warp (sct_register_to_template): anatomically accurate,
         non-linear, handles the straightened→curved mapping correctly.
      2. Seg-derived cord surface: always correctly aligned because it is built
         from the subject's own segmentation (subject-space → subject-space),
         never from PAM50 coordinates.

    Affine resampling of PAM50_cord.nii.gz directly into subject space is
    NOT used here — it produces a misaligned overlay because PAM50 is in
    MNI/ICBM152 space while subject T2w images are in scanner space.
    """
    native = _sct_native_pam50(subject, reference_path)
    if native and _nifti_compatible(native, reference_path):
        return native
    return _seg_based_cord_overlay(subject, reference_path)


def overlay_availability(subject: str, reference_path: str | None = None) -> dict[str, bool]:
    return {
        "seg": get_seg_path(subject, reference_path) is not None,
        "labels": get_labels_path(subject, reference_path) is not None,
        "pam50": get_seg_path(subject, reference_path) is not None if reference_path else False,
    }


def overlay_diagnostics(subject: str, reference_path: str | None = None) -> dict[str, str | bool | None]:
    seg = get_seg_path(subject, reference_path)
    labels = get_labels_path(subject, reference_path)
    pam50 = get_native_pam50_path(subject, reference_path) if reference_path else None

    any_labels = _find_file(
        subject,
        _LABELS_PATTERNS,
        reference_path=None,
        require_same_contrast=False,
    )
    labels_note = "ok" if labels else "missing"
    if not labels and any_labels:
        labels_note = f"no {_reference_contrast(reference_path) or 'matching'} labels; found {os.path.basename(any_labels)}"
    if labels and reference_path and not _has_world_overlap(labels, reference_path):
        labels_note = f"repo label does not overlap active anatomy: {os.path.basename(labels)}"
        labels = None

    return {
        "seg": seg,
        "labels": labels,
        "pam50": pam50,
        "labels_note": labels_note,
        "pam50_note": (
            "SCT-warped PAM50"
            if pam50 and "cord_surface_" not in os.path.basename(pam50)
            else "subject-fitted PAM50 cord proxy from native SC segmentation"
            if pam50
            else "missing native SC segmentation for subject-fitted PAM50 proxy"
        ),
    }


def build_overlay_volumes(
    subject: str,
    reference_path: str,
    include_seg: bool,
    include_labels: bool,
    include_pam50: bool,
) -> list[dict]:
    """Return anatomy-aware overlay volumes for ipyniivue multi-volume loading."""
    volumes = []
    ensure_subject_overlays(subject, reference_path)

    if include_seg:
        path = get_seg_path(subject, reference_path)
        if path and _has_world_overlap(path, reference_path):
            volumes.append({
                "path": path,
                "colormap": "red",
                "opacity": OVERLAY_OPACITY,
                "visible": True,
                "cal_min": 0.5,
                "cal_max": 1.0,
            })

    if include_labels:
        path = get_labels_path(subject, reference_path)
        if path and _has_world_overlap(path, reference_path):
            volumes.append({
                "path": path,
                "colormap": "warm",
                "opacity": OVERLAY_OPACITY,
                "visible": True,
                "cal_min": 1,
                "cal_max": len(VERTEBRAL_LEVELS),
            })

    if include_pam50:
        path = get_native_pam50_path(subject, reference_path)
        if path:
            volumes.append({
                "path": path,
                "colormap": "blue",
                "opacity": OVERLAY_OPACITY,
                "visible": True,
                "cal_min": 0.5,
                "cal_max": 1,
            })

    return volumes


def _repo_ready() -> bool:
    return os.path.isdir(os.path.join(_REPO_DIR, ".git"))


def _git_files_under(paths: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", *paths],
        cwd=_REPO_DIR,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        return []
    return [p.strip().replace("\\", "/") for p in result.stdout.splitlines() if p.strip()]


def _matching_repo_files(subject: str, patterns: tuple[str, ...]) -> list[str]:
    roots = [
        f"derivatives/labels/{subject}/anat",
        f"derivatives/labels_softseg/{subject}/anat",
        f"derivatives/labels_softseg_bin/{subject}/anat",
    ]
    repo_files = _git_files_under(roots)
    wanted = [pat.format(sub=subject).replace("\\", "/") for pat in patterns]
    matches = [
        path for path in repo_files
        if path.endswith((".nii", ".nii.gz")) and any(fnmatch(path, pat) for pat in wanted)
    ]
    t2w_matches = [p for p in matches if "_T2w" in os.path.basename(p)]
    matches = t2w_matches or matches
    return sorted(set(matches), key=lambda p: (0 if "manual" in p else 1, 0 if "_T2w" in p else 1, len(p), p))


def _download_repo_matches(subject: str, patterns: tuple[str, ...], label: str, bar=None) -> tuple[bool, str]:
    if not _repo_ready():
        return False, f"{label}: repository not cloned"
    paths = _matching_repo_files(subject, patterns)
    if not paths:
        return False, f"{label}: no matching files in repo"
    return _annex_get(paths, label, bar=bar)


def ensure_subject_overlays(subject: str, reference_path: str | None = None, bar=None) -> tuple[bool, str]:
    missing_seg = get_seg_path(subject, reference_path) is None
    missing_labels = get_labels_path(subject, reference_path) is None
    if not missing_seg and not missing_labels:
        return True, "ok"

    errors: list[str] = []
    if missing_seg:
        ok, msg = _download_repo_matches(subject, _SEG_TREE_PATTERNS, "seg (SC boundary)", bar=bar)
        if not ok:
            errors.append(msg)
    if missing_labels:
        ok, msg = _download_repo_matches(subject, _LABELS_TREE_PATTERNS, "labels (vertebral/disc levels)", bar=bar)
        if not ok:
            errors.append(msg)

    if errors:
        return False, "; ".join(errors)
    return True, "ok"


def _annex_get(paths: list[str], label: str, bar=None) -> tuple[bool, str]:
    if not paths:
        return False, f"{label}: no matching files in git tree"

    result = subprocess.run(
        ["git", "annex", "get", *paths],
        cwd=_REPO_DIR,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "git annex get failed"
        last = message.splitlines()[-1] if message else "git annex get failed"
        return False, f"{label}: {last}"

    _bar_update(bar, increment=20, subtitle=f"{label} - downloaded")
    return True, "ok"


def download_pam50(bar=None) -> tuple[bool, str]:
    """
    Download PAM50 files.

    Note: PAM50 files are only used when SCT (sct_register_to_template) is
    available.  Without SCT the cord overlay is derived from the subject's
    own segmentation and PAM50 files are not required.  We still download
    them so that SCT users get the full pipeline automatically.
    """
    os.makedirs(_PAM50_DIR, exist_ok=True)
    errors = []
    _bar_update(bar, increment=10, subtitle="Checking PAM50 template files...")
    for i, name in enumerate(_PAM50_DOWNLOADS, start=1):
        ok, msg = _download_pam50_file(name)
        if ok:
            _bar_update(bar, increment=25, subtitle=f"PAM50 {name} ready ({i}/{len(_PAM50_DOWNLOADS)})")
        else:
            errors.append(f"{name}: {msg}")

    if errors:
        return False, "; ".join(errors)

    path = get_pam50_path()
    _bar_update(bar, increment=15, subtitle="PAM50 files ready")
    return True, path or _PAM50_DIR


def download_subject_overlays(subject: str, bar=None) -> tuple[bool, str]:
    """
    Fetch spinal cord segmentation and vertebral/disc label NIfTIs for subject.

    Common spine-generic layout:
      derivatives/labels_softseg_bin/<sub>/anat/<sub>_T2w_desc-softseg_label-SC_seg.nii.gz
      derivatives/labels/<sub>/anat/<sub>_T2w_label-discs_dlabel.nii.gz
    """
    if not _repo_ready():
        return False, "Repository not cloned. Download a subject scan first."

    errors: list[str] = []
    emitted = 0

    def _emit(delta: int, subtitle: str) -> None:
        nonlocal emitted
        emitted += delta
        _bar_update(bar, increment=delta, subtitle=subtitle)

    _emit(5, "Checking repository...")

    targets = [
        ("seg (SC boundary)", _SEG_PATTERNS, _SEG_TREE_PATTERNS),
        ("labels (vertebral/disc levels)", _LABELS_PATTERNS, _LABELS_TREE_PATTERNS),
    ]

    for label, check_patterns, tree_patterns in targets:
        if _find_file(subject, check_patterns):
            _emit(25, f"{label} - already present")
            continue

        _emit(5, f"Finding {label} files...")
        annex_paths = _matching_repo_files(subject, tree_patterns)
        if not annex_paths:
            errors.append(f"{label}: no matching files found under derivatives for {subject}")
            continue

        _emit(5, f"Fetching {len(annex_paths)} {label} file(s) via git-annex...")
        try:
            ok, msg = _annex_get(annex_paths, label, bar=bar)
            emitted += 20 if ok else 0
            if not ok:
                errors.append(msg)
        except subprocess.TimeoutExpired:
            errors.append(f"{label}: timed out after 10 min")
        except Exception as exc:
            errors.append(f"{label}: {exc}")

    if errors:
        return False, "; ".join(errors)

    _bar_update(bar, increment=max(100 - emitted, 0), subtitle="All overlays ready")
    return True, "ok"
