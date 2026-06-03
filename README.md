# Spine MRI Viewer

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME)

![Marimo](https://img.shields.io/badge/Framework-Marimo-blue)
![Python](https://img.shields.io/badge/Python-3.12+-green)
![MRI](https://img.shields.io/badge/Domain-Medical%20Imaging-red)
![Status](https://img.shields.io/badge/Status-Ready-brightgreen)

> **Note:** Replace `YOUR_GITHUB_USERNAME/YOUR_REPO_NAME` in the Codespaces badge URL above with your actual GitHub repository path before publishing.

---

## Overview

The **Spine MRI Viewer** is an interactive medical imaging platform designed for multi-vendor MRI dataset exploration and cohort-level analysis.

It enables researchers to:
- Visualize MRI volumes interactively in 3D and multiplanar views
- Filter data by scanner vendor
- Export 3D views and metadata
- Perform cohort-level statistical comparisons

Built using **Marimo + ipyniivue**, optimized for browser-based execution and zero-install deployment.

---

## Quick Start (GitHub Codespaces)

Click the **Open in GitHub Codespaces** badge above. The environment will automatically:

1. Build a Python 3.12 container with all dependencies
2. Download the participants metadata
3. Launch the Marimo app on port 8080 and open it in your browser

No local installation required.

---

## Features

### Data Exploration
- Vendor filtering (GE / Philips / Siemens / Other)
- Subject-level selection
- Automatic dataset parsing

### MRI Visualization
- Interactive 3D/orthogonal MRI viewer (ipyniivue)
- T2-weighted volume rendering
- Slice-based exploration
- Auto-downsampling for large volumes (> 256³ voxels), with **Load Full Resolution** toggle

### Export Tools
- **Export 3D View PNG** — captures the live ipyniivue canvas directly in the browser
- **Export Slice PNG** — server-side fallback: central axial slice rendered with matplotlib
- **Export CSV** — subject metadata: ID, vendor, field strength, mean intensity, voxel dimensions

### Cohort Analysis
- Mean signal intensity per subject
- Standard deviation analysis
- Vendor-wise histogram/distribution plots (Plotly)

---

## Dataset Structure

The dataset follows the BIDS-compatible spine-generic format:

- Each subject contains an `anat/` folder
- T2-weighted MRI stored as compressed NIfTI (`.nii.gz`)
- Metadata stored in `participants.tsv` and per-subject JSON sidecars
- Large MRI files managed via **git-annex** (downloaded on demand)

---

## Important Note: Data Availability (Git Annex)

MRI files are stored via git-annex and downloaded on demand inside the app.

- `.nii.gz` files must be fully retrieved before visualization
- The app shows a download button per subject — click it to fetch the file
- Missing files are skipped automatically in cohort analysis

---

## Export Features

### 3D View PNG
Captures the current ipyniivue WebGL canvas using the browser's canvas API (`canvas.toDataURL`). Works for any view angle or rendering mode currently displayed. If the canvas is tainted (cross-origin data), the slice PNG fallback is available.

### Slice PNG
Generates the central axial slice of the T2 volume server-side via matplotlib. Always available once a subject is downloaded.

### CSV
Exports: `subject_id`, `vendor`, `field_strength`, `mean_intensity`, `voxel_dims`.

---

## Cohort Analysis

Provides statistical comparison across MRI vendors:

- **All vendors selected** — overlapping histograms of mean signal intensity, one per vendor
- **Single vendor selected** — distribution curve (distplot) for that vendor only

Subjects must be downloaded locally before they appear in the cohort plot.

---

## Performance

- Volumes with any axis > 256 voxels are automatically downsampled by factor 2 on first load
- A **Load Full Resolution** button is shown below the viewer — click to reload at full resolution
- Click again to return to the preview (downsampled) version
- Memory ceiling target: 2 GB

---

## Error Handling

The system gracefully handles:

- Missing MRI files
- Unavailable dataset entries
- Corrupted or incomplete NIfTI files
- Empty vendor filters
- Invalid subject selections

Fallback messages ensure stable UI behaviour throughout.

---

## Local Development

```bash
# Clone the repo
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Run the app
marimo run spine_app.py
```

Then open `http://localhost:2718` in your browser.

---

## Requirements

See `requirements.txt` for the full pinned dependency list. Key packages:

| Package | Purpose |
|---------|---------|
| `marimo` | Reactive notebook framework |
| `ipyniivue` | NIfTI volume viewer (WebGL) |
| `nibabel` | NIfTI file I/O |
| `numpy` | Numerical computation |
| `plotly` | Interactive cohort plots |
| `matplotlib` | Slice PNG export |
| `pandas` | Metadata and CSV handling |

---

## Summary

This project delivers a complete interactive MRI analysis system with:

- Interactive 3D visualization
- Cohort analysis across scanner vendors
- 3D view and metadata export
- GitHub Codespaces one-click deployment
- Automatic performance guardrails for large volumes
