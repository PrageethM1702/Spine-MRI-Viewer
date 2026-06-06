# Spine MRI Viewer

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PrageethM1702/Spine-MRI-Viewer)

![Marimo](https://img.shields.io/badge/Framework-Marimo-blue)
![Python](https://img.shields.io/badge/Python-3.12+-green)
![MRI](https://img.shields.io/badge/Domain-Spine%20MRI-red)
![Status](https://img.shields.io/badge/Status-Ready-brightgreen)

Interactive spine MRI viewer for the `spine-generic/data-multi-subject` dataset, built with Marimo, IPyNiiVue, Plotly, nibabel, and pandas.

## What It Does

- Filter participants by MRI vendor: GE, Philips, Siemens, Other, or All.
- Download subject T2w NIfTI files on demand from the spine-generic git-annex dataset.
- View anatomy in 2D multiplanar mode with aligned subject-specific overlays.
- View anatomy-aware 3D overlays using a Plotly world-space renderer.
- Display spinal cord boundary, vertebral/disc labels, and PAM50 cord/template proxy overlays.
- Export subject metadata CSV.
- Export 2D slice PNG and Plotly 3D PNG when Kaleido is installed.
- Upload metric CSV/TSV files and filter metrics by subject and vertebral level.
- Draw annotation arrows/text and download annotated screenshots.
- Measure distance, Cobb angle, and raw voxel intensity.

## Current Scope

Implemented:

- Vendor dropdown.
- Participant selector.
- Cohort histogram.
- Metadata CSV export.
- PNG export support.
- GitHub Codespaces configuration.
- Performance guardrails with Fast Preview and full-resolution overlay mode.
- Annotation and screenshot tools.
- Distance ruler and Cobb angle tool.
- Raw voxel intensity probe.
- Anatomy-aware 2D overlays.
- Anatomy-aware 3D visualization.
- Vertebral level labels.
- Spinal cord boundary overlays.
- PAM50 cord/template proxy overlay.
- Quantitative metrics panel.
- Metric CSV/TSV upload.

Paused by client scope:

- Offline PWA.
- DICOM upload.
- URL-based sharing.
- Case PDF export.
- Multi-user sync.

## Quick Start: GitHub Codespaces

Open the project in Codespaces:

[https://codespaces.new/PrageethM1702/Spine-MRI-Viewer](https://codespaces.new/PrageethM1702/Spine-MRI-Viewer)

The devcontainer will:

- Install Python dependencies from `requirements.txt`.
- Install `git-annex`.
- Create `data/cache`, `data/metrics`, and `data/data-multi-subject`.
- Download `participants.tsv`.
- Start Marimo on port `2718`.

## Local Setup

```bash
git clone https://github.com/PrageethM1702/Spine-MRI-Viewer.git
cd Spine-MRI-Viewer

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

marimo run spine_app.py
```

Open:

```text
http://localhost:2718
```

If your system does not open the browser automatically, copy the URL from the terminal.

## Metric CSV/TSV Upload

The quantitative metrics panel is ready, but metrics are not bundled with this repository by default.

Clients can upload CSA, FA, MTR, or related metric tables directly in the app. Uploaded files are saved under:

```text
data/metrics/
```

Recommended columns:

```text
participant_id, vertebral_level, CSA, FA, MTR
```

Accepted subject columns:

```text
participant_id, subject, subject_id, participant, sub
```

Accepted level columns:

```text
vertebral_level, level, label, vert_level, disc_level
```

If no metric files are available, the app shows an informational empty state instead of failing.

## Anatomical Overlays

The viewer uses native subject-space derivatives from the dataset when available:

```text
data/data-multi-subject/derivatives/labels/
data/data-multi-subject/derivatives/labels_softseg/
data/data-multi-subject/derivatives/labels_softseg_bin/
```

Supported overlays:

- Spinal cord segmentation/boundary: `_seg.nii.gz`.
- Vertebral or disc labels: `_labels.nii.gz`, `_dlabel.nii.gz`, label/disc derivative files.
- PAM50 cord/template proxy generated from subject-native segmentation when a real SCT warp is unavailable.

The 2D viewer uses IPyNiiVue multi-volume loading. The 3D view uses Plotly and world-space overlay geometry so the app does not depend on IPyNiiVue's render layout switching.

## Performance

The app has two modes:

- Full-resolution mode: recommended for anatomy-aware overlays.
- Fast Preview: loads a downsampled anatomy volume for speed and disables overlays to avoid alignment errors.

Use full-resolution mode when reviewing spinal cord boundary, vertebral labels, or PAM50 overlays.

## Export

Available exports:

- Subject metadata CSV.
- Metrics CSV for the selected subject/vertebral level.
- 2D slice PNG fallback.
- Current 3D Plotly PNG when `kaleido` is installed.

`requirements.txt` includes:

```text
kaleido==0.2.1
```

If PNG export is unavailable, run:

```bash
pip install -r requirements.txt
```

## Dataset Notes

Large NIfTI files are managed by git-annex in the upstream spine-generic dataset. The app downloads MRI and derivative files on demand.

Upstream dataset:

[https://github.com/spine-generic/data-multi-subject](https://github.com/spine-generic/data-multi-subject)

PAM50 reference:

[https://spinalcordtoolbox.com/stable/overview/concepts/pam50.html](https://spinalcordtoolbox.com/stable/overview/concepts/pam50.html)

## File Map

- `spine_app.py`: Marimo app UI.
- `viewer.py`: IPyNiiVue 2D/multivolume viewer.
- `overlays.py`: overlay discovery, download, and anatomy-aware validation.
- `anatomy_3d.py`: Plotly 3D spinal cord/label/PAM50 visualizer.
- `metrics.py`: metric CSV/TSV discovery, upload, and filtering.
- `analysis.py`: cohort analysis, downsampling, metadata export helpers.
- `measurement.py`: distance, angle, and raw intensity tools.
- `annotation.py`: 2D annotation and screenshot tools.

## Client Data Needed

To populate the quantitative metrics panel, provide CSV/TSV files containing precomputed CSA, FA, MTR, or related measures. These can be uploaded directly through the app.
