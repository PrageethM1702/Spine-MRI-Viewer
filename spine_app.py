import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    from data_loader import get_subjects_by_vendor, get_t2_path, get_vendor_for_subject
    from mri_downloader import is_cached, get_cached_path, download_subject
    from viewer import create_viewer
    from analysis import (
        export_t2_middle_slice_png,
        compute_vendor_intensity_summary,
        build_cohort_plotly_fig,
        get_subject_metrics,
        downsample_nifti_if_needed,
    )
    from annotation import AnnotationWidget
    from measurement import MeasurementWidget
    from anatomy_3d import build_anatomy_3d_figure, export_3d_figure_png
    from metrics import available_metric_tables, metrics_for_subject_level, save_metric_upload
    from overlays import (
        build_overlay_volumes,
        overlay_availability,
        overlay_diagnostics,
        VERTEBRAL_LEVELS,
        download_pam50,
        download_subject_overlays,
        get_seg_path,
        get_labels_path,
        get_pam50_path,
        get_native_pam50_path,
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import nibabel as nib
    import numpy as np
    import tempfile
    import json
    import os
    return (
        get_subjects_by_vendor, get_t2_path, get_vendor_for_subject,
        is_cached, get_cached_path, download_subject,
        create_viewer,
        export_t2_middle_slice_png, compute_vendor_intensity_summary,
        build_cohort_plotly_fig, get_subject_metrics, downsample_nifti_if_needed,
        AnnotationWidget,
        MeasurementWidget,
        build_anatomy_3d_figure, export_3d_figure_png,
        available_metric_tables, metrics_for_subject_level, save_metric_upload,
        build_overlay_volumes, overlay_availability, overlay_diagnostics, VERTEBRAL_LEVELS,
        download_pam50, download_subject_overlays,
        get_seg_path, get_labels_path, get_pam50_path, get_native_pam50_path,
        matplotlib, plt, pd, nib, np, tempfile, json, os,
    )


@app.cell
def _(mo):
    vendors = ["All", "GE", "Philips", "Siemens", "Other"]
    vendor = mo.ui.dropdown(vendors, value="All", label="Filter by Vendor")
    return (vendor,)


@app.cell
def _(mo, vendor, get_subjects_by_vendor):
    _subjects = get_subjects_by_vendor(vendor.value) or ["No data"]
    subject = mo.ui.dropdown(_subjects, value=_subjects[0], label="Select Subject")
    return (subject,)


@app.cell
def _(mo, subject):
    download_btn = mo.ui.button(
        label=f"Download {subject.value}",
        value=0,
        on_click=lambda v: v + 1,
    )
    return (download_btn,)


@app.cell
def _(
    mo, vendor, subject, download_btn, is_cached, get_cached_path,
    download_subject, download_subject_overlays, download_pam50, os,
):
    if download_btn.value > 0 and not is_cached(subject.value):
        with mo.status.progress_bar(
            total=100,
            title=f"Downloading {subject.value}",
            subtitle="Starting download...",
            completion_title="Download complete",
            show_rate=False,
            show_eta=False,
            remove_on_exit=False,
        ) as _bar:
            _bar.update(increment=5, subtitle="Cloning repository (first run may take a moment)...")
            _dl = download_subject(subject.value)
            if _dl and os.path.exists(_dl) and os.path.getsize(_dl) > 1024:
                _bar.update(increment=35, subtitle=f"MRI done - {os.path.getsize(_dl) / 1024 / 1024:.1f} MB")
                download_subject_overlays(subject.value, bar=_bar)
                download_pam50(bar=_bar)
            else:
                _bar.update(increment=95, subtitle="Failed - check that git-annex is installed")
    elif download_btn.value > 0 and is_cached(subject.value):
        with mo.status.progress_bar(
            total=100,
            title=f"Downloading overlays for {subject.value}",
            subtitle="Fetching missing overlay files...",
            completion_title="Overlay download complete",
            show_rate=False,
            show_eta=False,
            remove_on_exit=False,
        ) as _bar:
            download_subject_overlays(subject.value, bar=_bar)
            download_pam50(bar=_bar)

    if is_cached(subject.value):
        _status = mo.callout(mo.md(f"Subject **{subject.value}** is cached and ready."), kind="success")
    else:
        _status = mo.callout(
            mo.md(f"Subject **{subject.value}** not downloaded. Click below to fetch."),
            kind="info",
        )

    result_path = get_cached_path(subject.value) if is_cached(subject.value) else None

    mo.vstack([vendor, subject, _status, download_btn])
    return (result_path,)


@app.cell
def _(mo, subject, result_path, get_t2_path):
    _path = get_t2_path(subject.value) or result_path
    if _path:
        hd_btn = mo.ui.button(label="Load Full Resolution", value=0, on_click=lambda v: v + 1)
        fast_preview = mo.ui.checkbox(value=False, label="Fast preview")
        view_mode = mo.ui.radio(
            options=["2D multiplanar", "3D render"],
            value="2D multiplanar",
            label="View Mode",
        )
    else:
        hd_btn = None
        fast_preview = None
        view_mode = None
    return (hd_btn, fast_preview, view_mode)


@app.cell
def _(
    mo, subject, result_path, get_t2_path, create_viewer, downsample_nifti_if_needed,
    hd_btn, fast_preview, view_mode, build_anatomy_3d_figure,
    build_overlay_volumes, overlay_availability, overlay_diagnostics,
    get_seg_path, get_labels_path, get_native_pam50_path, os,
):
    from ipyniivue import SliceType as _SliceType
    _path = get_t2_path(subject.value) or result_path
    active_nv = None
    current_3d_fig = None
    selected_level_from_3d = None

    if _path and hd_btn is not None and fast_preview is not None and view_mode is not None and hd_btn.value >= 0:
        try:
            _use_full_res = hd_btn.value > 0
            _is_fast_preview = bool(fast_preview.value) and not _use_full_res
            _load_path = downsample_nifti_if_needed(_path, threshold=256) if _is_fast_preview else _path
            _res_note = mo.callout(
                mo.md(
                    "Fast preview loaded without anatomy overlays. Click **Load Full Resolution** or turn off **Fast preview** for aligned overlays."
                    if _is_fast_preview
                    else "Viewing at **full resolution** so derivative overlays use the native anatomy grid."
                ),
                kind="warn" if _is_fast_preview else "success",
            )

            # Auto-detect which overlays are available and include them all
            _avail = overlay_availability(subject.value, reference_path=_path)
            _diag = overlay_diagnostics(subject.value, reference_path=_path)
            _ovl = [] if _is_fast_preview else build_overlay_volumes(
                subject.value, _path,
                include_seg=_avail["seg"], include_labels=_avail["labels"], include_pam50=_avail["pam50"],
            )

            _is_3d = view_mode.value == "3D render"
            _slice_type = _SliceType.RENDER if _is_3d else _SliceType.MULTIPLANAR
            if _is_3d:
                active_nv = None
                current_3d_fig = build_anatomy_3d_figure(
                    _path,
                    seg_path=None if _is_fast_preview else get_seg_path(subject.value, _path),
                    labels_path=None if _is_fast_preview else get_labels_path(subject.value, _path),
                    pam50_path=None if _is_fast_preview else get_native_pam50_path(subject.value, _path),
                )
                _viewer_widget = mo.ui.plotly(current_3d_fig)
            else:
                active_nv = create_viewer(
                    _load_path,
                    overlay_volumes=_ovl,
                    slice_type=_slice_type,
                    show_render=False,
                )
                _viewer_widget = active_nv
            _items = [
                mo.md(f"### Viewing: `{subject.value}`"),
            ]
            if _res_note:
                _items.append(_res_note)
            _notes = []
            if _diag.get("labels_note") != "ok":
                _notes.append(f"Labels: {_diag.get('labels_note')}")
            if not _diag.get("pam50"):
                _notes.append(f"PAM50: {_diag.get('pam50_note')}")
            if _notes:
                _items.append(mo.callout(mo.md("  \n".join(_notes)), kind="warn"))
            _items.append(fast_preview)
            _items.append(view_mode)
            _items.append(_viewer_widget)
            _items.append(hd_btn)
            _viewer_out = mo.vstack(_items)
        except Exception as _e:
            _viewer_out = mo.callout(mo.md(f"Viewer error: `{_e}`"), kind="danger")
    else:
        _viewer_out = mo.md("_Download a subject to view it here._")

    _viewer_out
    return (active_nv, current_3d_fig, selected_level_from_3d)


@app.cell
def _(mo, result_path, get_t2_path, subject, get_vendor_for_subject,
      get_subject_metrics, export_t2_middle_slice_png, export_3d_figure_png,
      pd, active_nv, current_3d_fig):
    _path = get_t2_path(subject.value) or result_path

    if _path:
        _vendor_str = get_vendor_for_subject(subject.value)
        _metrics    = get_subject_metrics(subject.value, _path, _vendor_str)
        _df         = pd.DataFrame([_metrics])
        _fallback_png = export_t2_middle_slice_png(_path)
        _save_filename = f"{subject.value}_3d_view.png"

        def _do_save_scene(_v):
            if active_nv is not None:
                active_nv.save_scene(_save_filename)
            return _v + 1

        _export_items = [mo.md("### Export")]
        if current_3d_fig is not None:
            _plotly_png = export_3d_figure_png(current_3d_fig)
            if _plotly_png:
                _export_items.append(mo.download(
                    data=_plotly_png,
                    filename=f"{subject.value}_3d_view.png",
                    mimetype="image/png",
                    label="Export Current 3D PNG",
                ))
            else:
                _export_items.append(mo.callout(
                    mo.md("3D PNG export requires `kaleido` in the runtime. CSV and 2D fallback export are still available."),
                    kind="warn",
                ))
        else:
            _export_3d_btn = mo.ui.button(
                label="Export 3D View PNG",
                value=0,
                on_click=_do_save_scene,
            )
            _export_items.append(_export_3d_btn)

        _btns = [
            mo.download(
                data=_df.to_csv(index=False).encode(),
                filename=f"{subject.value}_metadata.csv",
                mimetype="text/csv",
                label="Export CSV",
            )
        ]
        if _fallback_png:
            _btns.append(mo.download(
                data=_fallback_png,
                filename=f"{subject.value}_slice.png",
                mimetype="image/png",
                label="Export Slice PNG (2D fallback)",
            ))
        _export_items.append(mo.hstack(_btns))
        _export_out = mo.vstack(_export_items)
    else:
        _export_out = mo.md("_Export available after download._")

    _export_out


@app.cell
def _(mo, subject, result_path, get_t2_path, selected_level_from_3d, VERTEBRAL_LEVELS):
    _path = get_t2_path(subject.value) or result_path
    if _path:
        _level_options = ["All levels", *VERTEBRAL_LEVELS]
        _default_level = selected_level_from_3d if selected_level_from_3d in VERTEBRAL_LEVELS else "All levels"
        metric_level = mo.ui.dropdown(_level_options, value=_default_level, label="Metric Level")
    else:
        metric_level = None
    return (metric_level,)


@app.cell
def _(mo):
    metric_upload = mo.ui.file(
        filetypes=[".csv", ".tsv"],
        multiple=True,
        label="Upload Metric CSV/TSV",
    )
    return (metric_upload,)


@app.cell
def _(mo, metric_upload, save_metric_upload):
    _saved = []
    for _file in metric_upload.value or []:
        _path = save_metric_upload(_file.name, _file.contents)
        if _path:
            _saved.append(_path)
    if _saved:
        upload_status = mo.callout(
            mo.md("Uploaded metric table(s):  \n" + "  \n".join(f"- `{p}`" for p in _saved)),
            kind="success",
        )
    else:
        upload_status = None
    return (upload_status,)


@app.cell
def _(mo, subject, result_path, get_t2_path, selected_level_from_3d,
      metric_level, metric_upload, upload_status, available_metric_tables, metrics_for_subject_level):
    _path = get_t2_path(subject.value) or result_path
    if not _path or metric_level is None:
        _metrics_out = mo.md("_Metrics appear after a subject is downloaded._")
    else:
        _level_value = None if metric_level.value == "All levels" else metric_level.value
        _df, _msg = metrics_for_subject_level(subject.value, _level_value)
        _tables = available_metric_tables()

        _items = [mo.md("### Quantitative Metrics")]
        if selected_level_from_3d:
            _items.append(mo.callout(mo.md(f"3D selected level: **{selected_level_from_3d}**"), kind="info"))
        _items.append(metric_upload)
        if upload_status:
            _items.append(upload_status)
        _items.append(metric_level)
        if not _tables:
            _items.append(mo.callout(
                mo.md("Quantitative metrics panel is ready. Add CSA/FA/MTR CSV or TSV files under `data/` to populate it."),
                kind="info",
            ))
        elif _df.empty:
            _items.append(mo.callout(mo.md(_msg), kind="warn"))
        else:
            _items.append(mo.ui.table(_df.head(50), page_size=10))
            _items.append(mo.download(
                data=_df.to_csv(index=False).encode(),
                filename=f"{subject.value}_{metric_level.value.replace(' ', '_')}_metrics.csv",
                mimetype="text/csv",
                label="Export Metrics CSV",
            ))
        _metrics_out = mo.vstack(_items)
    _metrics_out


@app.cell
def _(mo, vendor, result_path, build_cohort_plotly_fig):
    _ = result_path
    _fig = build_cohort_plotly_fig(vendor.value)
    if _fig is not None:
        _cohort_out = mo.vstack([mo.md("### Cohort Analysis"), mo.ui.plotly(_fig)])
    else:
        _cohort_out = mo.md("_Cohort analysis appears after at least one subject is downloaded._")
    _cohort_out


@app.cell
def _(mo, subject, result_path, get_t2_path, AnnotationWidget):
    _path = get_t2_path(subject.value) or result_path
    if _path:
        ann_widget = mo.ui.anywidget(AnnotationWidget(_path))
    else:
        ann_widget = None
    return (ann_widget,)


@app.cell
def _(mo, subject, ann_widget):
    if ann_widget is None:
        _ann_out = mo.vstack([
            mo.md("---\n## Annotation"),
            mo.md("_Download a subject to use the annotation tool._"),
        ])
    else:
        _ = ann_widget.value  # subscribe to state changes
        _ann_out = mo.vstack([
            mo.md("---\n## Annotation"),
            mo.md("### Annotate 2D Slice"),
            mo.md(
                "Click to set arrow **start**, click again to set arrow **end**. "
                "Use **Undo** / **Clear** to edit."
            ),
            ann_widget,
            mo.download(
                data=ann_widget.widget.png_bytes,
                filename=f"{subject.value}_annotated.png",
                mimetype="image/png",
                label="Download Annotated Screenshot",
            ),
        ])
    _ann_out


@app.cell
def _(mo, subject, result_path, get_t2_path, MeasurementWidget):
    _path = get_t2_path(subject.value) or result_path
    if _path:
        meas_widget = mo.ui.anywidget(MeasurementWidget(_path))
    else:
        meas_widget = None
    return (meas_widget,)


@app.cell
def _(mo, subject, meas_widget):
    if meas_widget is None:
        _meas_out = mo.vstack([
            mo.md("---\n## Measurement Tools"),
            mo.md("_Download a subject to use the measurement tools._"),
        ])
    else:
        _ = meas_widget.value  # subscribe to state changes
        _last = meas_widget.widget.last_results
        _results_md = ""
        if _last:
            _rows = "\n".join(
                f"| {i+1} | {r.get('value', r):.4f} |"
                for i, r in enumerate(_last)
                if isinstance(r, dict) and "value" in r
            )
            if _rows:
                _results_md = (
                    "\n\n**Last intensity probe results:**\n\n"
                    "| # | Raw Value |\n|---|---|\n" + _rows
                )
        _meas_out = mo.vstack([
            mo.md("---\n## Measurement Tools"),
            mo.md(
                "**Ruler (mm):** click 2 points → Euclidean distance.  \n"
                "**Cobb Angle (°):** click 3 points (endplate A → apex → endplate B).  \n"
                "**Intensity Probe:** click 1 point → raw NIfTI voxel value "
                "(HU for CT, relative intensity for MRI)."
                + _results_md
            ),
            meas_widget,
        ])
    _meas_out


@app.cell
def _(mo):
    mo.md("---\n## Upload Your Own MRI\nUpload a `.nii` or `.nii.gz` file to visualize it.")


@app.cell
def _(mo):
    file_upload = mo.ui.file(filetypes=[".nii", ".nii.gz"], label="Upload NIfTI file")
    file_upload
    return (file_upload,)


@app.cell
def _(mo, file_upload, create_viewer, export_t2_middle_slice_png,
      downsample_nifti_if_needed, nib, np, plt, tempfile):
    if not file_upload.value:
        _upload_out = mo.callout(mo.md("Upload a file above to begin."), kind="info")
    else:
        _up    = file_upload.value[0]
        _fname = _up.name
        _fbytes = _up.contents
        _suffix = ".nii.gz" if _fname.endswith(".nii.gz") else ".nii"
        _tmp = tempfile.NamedTemporaryFile(delete=False, suffix=_suffix)
        _tmp.write(_fbytes)
        _tmp.flush()
        _upath     = _tmp.name
        _load_path = downsample_nifti_if_needed(_upath, threshold=256)
        _sname     = _fname.replace(".nii.gz", "").replace(".nii", "")

        _out = [mo.md(f"Loaded: `{_fname}` ({len(_fbytes)/1024/1024:.1f} MB)")]
        if _load_path != _upath:
            _out.append(mo.callout(mo.md("Loaded at half resolution (volume > 256³)."), kind="warn"))

        try:
            _out.append(mo.vstack([mo.md("### MRI Viewer"), create_viewer(_load_path)]))
        except Exception as _e:
            _out.append(mo.callout(mo.md(f"Viewer error: `{_e}`"), kind="danger"))

        _png = export_t2_middle_slice_png(_upath)
        if _png:
            _out.append(mo.vstack([
                mo.md("### Export"),
                mo.download(data=_png, filename=f"{_sname}_slice.png", mimetype="image/png", label="Export PNG (middle slice)"),
            ]))

        try:
            _data  = nib.load(_upath).get_fdata()
            _zooms = nib.load(_upath).header.get_zooms()[:3]
            _fig4, _ax4 = plt.subplots()
            _ax4.hist(_data.flatten(), bins=60, color="steelblue", alpha=0.8)
            _ax4.set_title("Intensity Distribution")
            _ax4.set_xlabel("Intensity")
            _ax4.set_ylabel("Voxel Count")
            _out.append(mo.vstack([mo.md(f"""### Analysis
| Property | Value |
|----------|-------|
| Shape | `{_data.shape}` |
| Voxel dims | `{_zooms[0]:.2f} x {_zooms[1]:.2f} x {_zooms[2]:.2f} mm` |
| Mean intensity | `{float(np.mean(_data)):.2f}` |
| Std intensity | `{float(np.std(_data)):.2f}` |
"""), _fig4]))
        except Exception as _e:
            _out.append(mo.callout(mo.md(f"Analysis failed: `{_e}`"), kind="danger"))

        _upload_out = mo.vstack(_out)
    _upload_out


if __name__ == "__main__":
    app.run()
