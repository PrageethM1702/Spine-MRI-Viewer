"""
measurement.py — Feature 14 + New C

Feature 14:
  - Distance ruler: click 2 points → Euclidean distance in mm
    using pixdim from NIfTI header (img.header.get_zooms())
  - Angle / Cobb tool: click 3 points → angle in degrees

New C:
  - Voxel intensity probe: click 1 point → raw NIfTI array value
    at those voxel coordinates (works for MRI relative intensity
    and CT Hounsfield Units — no normalisation applied)

All three tools share one anywidget canvas so the slice controls
are consistent with annotation.py.  The active tool is switched
via a toolbar button row.  Results are displayed in a floating
panel below the image and also synced back to Python via the
`results` traitlet (JSON list) for downstream use.
"""
import io
import json
import base64
import math
import anywidget
import traitlets
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



def _get_slice(nifti_path: str, axis: str, index: int | None):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    if axis == "axial":
        max_idx = data.shape[2] - 1
        idx = index if index is not None else data.shape[2] // 2
        idx = max(0, min(idx, max_idx))
        return data[:, :, idx].T, max_idx, idx
    elif axis == "sagittal":
        max_idx = data.shape[0] - 1
        idx = index if index is not None else data.shape[0] // 2
        idx = max(0, min(idx, max_idx))
        return data[idx, :, :].T, max_idx, idx
    else:
        max_idx = data.shape[1] - 1
        idx = index if index is not None else data.shape[1] // 2
        idx = max(0, min(idx, max_idx))
        return data[:, idx, :].T, max_idx, idx


def _render_png(nifti_path: str, axis: str, index: int | None,
                fig_size: int = 500) -> tuple[bytes, int, int]:
    slice_data, max_idx, used_idx = _get_slice(nifti_path, axis, index)
    dpi = 100
    size_in = fig_size / dpi
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
    ax.imshow(slice_data, cmap="gray", origin="upper", aspect="equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), max_idx, used_idx


def _get_voxel_value(nifti_path: str, axis: str, slice_idx: int,
                     norm_x: float, norm_y: float) -> float:
    """
    Return the raw NIfTI array value at normalised canvas coordinates.
    norm_x, norm_y are in [0,1] relative to the displayed image.
    The slice orientation transpose matches _get_slice above.
    """
    img  = nib.load(nifti_path)
    data = img.get_fdata()

    if axis == "axial":
        h, w = data.shape[1], data.shape[0]
        # slice: data[:,:,idx].T  →  rows=y(dim1), cols=x(dim0)
        xi = int(norm_x * w)
        yi = int(norm_y * h)
        xi = max(0, min(xi, w - 1))
        yi = max(0, min(yi, h - 1))
        return float(data[xi, yi, slice_idx])

    elif axis == "sagittal":
        h, w = data.shape[2], data.shape[1]
        # slice: data[idx,:,:].T  →  rows=z(dim2), cols=y(dim1)
        xi = int(norm_x * w)
        yi = int(norm_y * h)
        xi = max(0, min(xi, w - 1))
        yi = max(0, min(yi, h - 1))
        return float(data[slice_idx, xi, yi])

    else:  # coronal
        h, w = data.shape[2], data.shape[0]
        # slice: data[:,idx,:].T  →  rows=z(dim2), cols=x(dim0)
        xi = int(norm_x * w)
        yi = int(norm_y * h)
        xi = max(0, min(xi, w - 1))
        yi = max(0, min(yi, h - 1))
        return float(data[xi, slice_idx, yi])


def _get_pixdim(nifti_path: str, axis: str) -> tuple[float, float]:
    """
    Return (px_mm, py_mm) — the mm per pixel for the two in-plane axes
    of the displayed slice.
    zooms order: (dim0, dim1, dim2, ...) = (x, y, z, ...)
    """
    zooms = nib.load(nifti_path).header.get_zooms()
    zx, zy, zz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    if axis == "axial":
        return zx, zy      # cols=x, rows=y
    elif axis == "sagittal":
        return zy, zz      # cols=y, rows=z
    else:
        return zx, zz      # cols=x, rows=z



class MeasurementWidget(anywidget.AnyWidget):
    """
    Three-tool measurement widget.

    Traitlets (Python → JS):
        img_b64   — base64 PNG of the current slice
        max_idx   — slider max
        init_idx  — starting slice index
        init_axis — starting axis
        px_mm_x   — mm per pixel, horizontal axis
        px_mm_y   — mm per pixel, vertical axis

    Traitlets (JS → Python):
        state     — JSON: {"axis", "slice", "tool", "points": [...]}
        results   — JSON list of result dicts for downstream use

    Python properties:
        last_results — parsed list of result dicts
    """

    img_b64   = traitlets.Unicode("").tag(sync=True)
    max_idx   = traitlets.Int(0).tag(sync=True)
    init_idx  = traitlets.Int(0).tag(sync=True)
    init_axis = traitlets.Unicode("axial").tag(sync=True)
    px_mm_x   = traitlets.Float(1.0).tag(sync=True)
    px_mm_y   = traitlets.Float(1.0).tag(sync=True)

    state   = traitlets.Unicode("{}").tag(sync=True)
    results = traitlets.Unicode("[]").tag(sync=True)

    _nifti_path: str = ""

    _esm = r"""
function render({ model, el }) {
  const FIG = 500;
  let POINTS  = [];   // accumulated click points for current measurement
  let HISTORY = [];   // completed measurements [{type, label, points}]
  let TOOL    = "distance";   // "distance" | "angle" | "intensity"

  // DOM 
  el.style.fontFamily = "sans-serif";
  el.style.userSelect = "none";

  // Toolbar
  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;gap:6px;align-items:center;margin-bottom:8px;flex-wrap:wrap";

  function toolBtn(label, tool, title) {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = title;
    b.dataset.tool = tool;
    b.style.cssText = "padding:4px 12px;font-size:13px;border-radius:4px;border:1px solid #ccc;cursor:pointer;background:#fff;transition:background 0.15s";
    b.addEventListener("click", () => setTool(tool));
    return b;
  }

  const btnDist  = toolBtn("Ruler (mm)",     "distance",  "Click 2 points to measure distance");
  const btnAngle = toolBtn("Cobb Angle (°)", "angle",     "Click 3 points: A, apex, B to measure Cobb angle");
  const btnProbe = toolBtn("Intensity Probe","intensity", "Click 1 point to read raw voxel value");
  const btnClear = document.createElement("button");
  btnClear.textContent = "Clear All";
  btnClear.style.cssText = "padding:4px 10px;font-size:13px;border-radius:4px;border:1px solid #e55;color:#c00;cursor:pointer;background:#fff;margin-left:8px";
  btnClear.addEventListener("click", clearAll);

  toolbar.append(btnDist, btnAngle, btnProbe, btnClear);

  // Slice controls
  const controls = document.createElement("div");
  controls.style.cssText = "display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap";

  const axisLabel = document.createElement("span");
  axisLabel.textContent = "Axis:";
  axisLabel.style.cssText = "font-size:13px;color:#666";

  const axisSelect = document.createElement("select");
  axisSelect.style.cssText = "font-size:13px;padding:3px 8px;border-radius:4px;border:1px solid #ccc";
  ["axial","sagittal","coronal"].forEach(v => {
    const o = document.createElement("option");
    o.value = o.textContent = v;
    axisSelect.appendChild(o);
  });

  const sliceLabel = document.createElement("span");
  sliceLabel.style.cssText = "font-size:13px;color:#666";
  const sliceNum = document.createElement("strong");
  const maxSpan  = document.createElement("span");
  sliceLabel.append("Slice: ", sliceNum, " / ", maxSpan);

  const sliceRange = document.createElement("input");
  sliceRange.type = "range";
  sliceRange.min  = 0;
  sliceRange.style.width = "140px";

  controls.append(axisLabel, axisSelect, sliceLabel, sliceRange);

  // Canvas wrapper
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "position:relative;display:inline-block;border:1px solid #ddd;border-radius:4px;overflow:hidden;line-height:0";

  const img    = document.createElement("img");
  img.style.cssText = `display:block;width:${FIG}px;height:${FIG}px`;
  img.draggable = false;

  const canvas = document.createElement("canvas");
  canvas.width  = FIG;
  canvas.height = FIG;
  canvas.style.cssText = `position:absolute;top:0;left:0;width:${FIG}px;height:${FIG}px;cursor:crosshair`;

  wrapper.append(img, canvas);

  // Results panel
  const panel = document.createElement("div");
  panel.style.cssText = "margin-top:10px;padding:10px;background:#f8f8f8;border:1px solid #ddd;border-radius:6px;font-size:13px;min-height:36px;font-family:monospace";
  panel.textContent = "No measurements yet.";

  // Hint
  const hint = document.createElement("div");
  hint.style.cssText = "margin-top:6px;font-size:12px;color:#888";

  el.append(toolbar, controls, wrapper, panel, hint);

  const ctx = canvas.getContext("2d");

  // Tool switching 
  function setTool(t) {
    TOOL    = t;
    POINTS  = [];
    redraw();
    updateHint();
    highlightBtn();
  }

  function highlightBtn() {
    [btnDist, btnAngle, btnProbe].forEach(b => {
      b.style.background = b.dataset.tool === TOOL ? "#ddeeff" : "#fff";
      b.style.fontWeight  = b.dataset.tool === TOOL ? "600" : "400";
    });
  }

  function updateHint() {
    const needed = TOOL === "distance" ? 2 : TOOL === "angle" ? 3 : 1;
    const rem = needed - POINTS.length;
    if (TOOL === "distance")
      hint.textContent = rem === 2 ? "Click point A." : "Click point B.";
    else if (TOOL === "angle")
      hint.textContent = rem === 3 ? "Click vertebral endplate A." :
                         rem === 2 ? "Click apex." : "Click vertebral endplate B.";
    else
      hint.textContent = "Click any point to read its raw voxel value.";
  }

  // Model init 
  function initFromModel() {
    sliceRange.max   = model.get("max_idx");
    sliceRange.value = model.get("init_idx");
    maxSpan.textContent  = model.get("max_idx");
    sliceNum.textContent = model.get("init_idx");
    axisSelect.value = model.get("init_axis");
    const b64 = model.get("img_b64");
    if (b64) img.src = "data:image/png;base64," + b64;
    highlightBtn();
    updateHint();
  }

  model.on("change:img_b64", () => {
    const b64 = model.get("img_b64");
    if (b64) img.src = "data:image/png;base64," + b64;
    POINTS = [];
    redraw();
  });
  model.on("change:max_idx", () => {
    sliceRange.max = model.get("max_idx");
    maxSpan.textContent = model.get("max_idx");
  });

  // Push state.
  // completedProbePoint: pass the finalised {x,y} for intensity so Python
  // receives it even after POINTS has already been cleared.
  function pushState(completedProbePoint) {
    const payload = {
      axis:   axisSelect.value,
      slice:  parseInt(sliceRange.value),
      tool:   TOOL,
      points: POINTS,
    };
    if (completedProbePoint) payload.intensity_probe = completedProbePoint;
    model.set("state", JSON.stringify(payload));
    model.save_changes();
  }

  // Slice / axis controls 
  axisSelect.addEventListener("change", () => {
    POINTS = []; HISTORY = [];
    panel.textContent = "No measurements yet.";
    pushState();
  });

  sliceRange.addEventListener("input", () => {
    sliceNum.textContent = sliceRange.value;
    POINTS = []; HISTORY = [];
    panel.textContent = "No measurements yet.";
    pushState();
  });

  // Canvas click 
  canvas.addEventListener("click", e => {
    const r  = canvas.getBoundingClientRect();
    const nx = (e.clientX - r.left) / r.width;
    const ny = (e.clientY - r.top)  / r.height;
    POINTS.push({ x: nx, y: ny });
    redraw();

    const needed = TOOL === "distance" ? 2 : TOOL === "angle" ? 3 : 1;
    if (POINTS.length >= needed) {
      finaliseMeasurement();
    } else {
      updateHint();
    }
  });

  // Finalise 
  function finaliseMeasurement() {
    const px = model.get("px_mm_x");
    const py = model.get("px_mm_y");
    let label = "";
    let probePoint = null;

    if (TOOL === "distance") {
      const dx = (POINTS[1].x - POINTS[0].x) * FIG * px;
      const dy = (POINTS[1].y - POINTS[0].y) * FIG * py;
      const dist = Math.sqrt(dx*dx + dy*dy);
      label = `Distance: ${dist.toFixed(2)} mm`;
    } else if (TOOL === "angle") {
      label = `Cobb angle: ${calcCobb(POINTS).toFixed(1)}°`;
    } else {
      // Intensity — Python resolves the exact value.
      // Capture the point NOW before POINTS is cleared below.
      probePoint = { ...POINTS[0] };
      label = "Intensity: (fetching...)";
    }

    HISTORY.push({ type: TOOL, label, points: [...POINTS] });
    POINTS = [];
    redraw();
    renderPanel();
    updateHint();
    pushState(probePoint);   // probePoint is non-null only for intensity
  }

  function calcCobb(pts) {
    // pts[0]=A, pts[1]=apex, pts[2]=B
    // Cobb angle = angle at apex between vectors apex→A and apex→B
    const ax = (pts[0].x - pts[1].x) * FIG;
    const ay = (pts[0].y - pts[1].y) * FIG;
    const bx = (pts[2].x - pts[1].x) * FIG;
    const by = (pts[2].y - pts[1].y) * FIG;
    const dot  = ax*bx + ay*by;
    const magA = Math.sqrt(ax*ax + ay*ay);
    const magB = Math.sqrt(bx*bx + by*by);
    if (magA < 1 || magB < 1) return 0;
    return (Math.acos(Math.max(-1, Math.min(1, dot/(magA*magB)))) * 180) / Math.PI;
  }

  // Results panel 
  function renderPanel() {
    if (!HISTORY.length) { panel.textContent = "No measurements yet."; return; }
    panel.innerHTML = HISTORY.map((m, i) =>
      `<div style="margin:2px 0"><span style="color:#888">#${i+1}</span> ${m.label}</div>`
    ).join("");
  }

  // Sync results to Python 
  model.on("change:results", () => {
    try {
      const res = JSON.parse(model.get("results"));
      // results only contains intensity entries; match to HISTORY by coords
      res.forEach(r => {
        if (r.value === undefined) return;
        const entry = HISTORY.find(m =>
          m.type === "intensity" &&
          m.points.length > 0 &&
          Math.abs(m.points[0].x - r.x) < 0.001 &&
          Math.abs(m.points[0].y - r.y) < 0.001
        );
        if (entry) entry.label = `Intensity: ${r.value.toFixed(4)}`;
      });
      renderPanel();
    } catch(e) {}
  });

  // Drawing 
  const COLORS = { distance: "#2979FF", angle: "#00C853", intensity: "#FF6D00" };

  function redraw() {
    ctx.clearRect(0, 0, FIG, FIG);

    HISTORY.forEach(m => {
      ctx.strokeStyle = COLORS[m.type] || "#999";
      ctx.fillStyle   = COLORS[m.type] || "#999";
      drawMeasurement(m.type, m.points);
    });

    // In-progress points
    if (POINTS.length > 0) {
      ctx.strokeStyle = "#aaa";
      ctx.fillStyle   = "#aaa";
      POINTS.forEach(p => drawDot(p.x*FIG, p.y*FIG));
      if (POINTS.length === 2 && TOOL === "angle") {
        drawLine(POINTS[0].x*FIG, POINTS[0].y*FIG,
                 POINTS[1].x*FIG, POINTS[1].y*FIG);
      }
    }
  }

  function drawMeasurement(type, pts) {
    if (type === "distance" && pts.length >= 2) {
      drawLine(pts[0].x*FIG, pts[0].y*FIG, pts[1].x*FIG, pts[1].y*FIG);
      drawDot(pts[0].x*FIG, pts[0].y*FIG);
      drawDot(pts[1].x*FIG, pts[1].y*FIG);
    } else if (type === "angle" && pts.length >= 3) {
      drawLine(pts[0].x*FIG, pts[0].y*FIG, pts[1].x*FIG, pts[1].y*FIG);
      drawLine(pts[1].x*FIG, pts[1].y*FIG, pts[2].x*FIG, pts[2].y*FIG);
      pts.forEach(p => drawDot(p.x*FIG, p.y*FIG));
    } else if (type === "intensity" && pts.length >= 1) {
      drawCrosshair(pts[0].x*FIG, pts[0].y*FIG);
    }
  }

  function drawLine(x0, y0, x1, y1) {
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
  }

  function drawDot(x, y) {
    ctx.beginPath(); ctx.arc(x, y, 5, 0, 2*Math.PI); ctx.fill();
  }

  function drawCrosshair(x, y) {
    ctx.lineWidth = 1.5;
    const s = 10;
    ctx.beginPath();
    ctx.moveTo(x-s, y); ctx.lineTo(x+s, y);
    ctx.moveTo(x, y-s); ctx.lineTo(x, y+s);
    ctx.stroke();
    drawDot(x, y);
  }

  // Clear 
  function clearAll() {
    POINTS = []; HISTORY = [];
    ctx.clearRect(0, 0, FIG, FIG);
    panel.textContent = "No measurements yet.";
    model.set("results", "[]");
    model.save_changes();
    updateHint();
  }

  initFromModel();
}
export default { render };
"""

    _css = """
:host { display: block; padding: 8px 0; }
"""

    def __init__(self, nifti_path: str, axis: str = "axial",
                 slice_index: int | None = None, **kwargs):
        self._nifti_path = nifti_path
        png_bytes, max_idx, used_idx = _render_png(nifti_path, axis, slice_index)
        px_x, px_y = _get_pixdim(nifti_path, axis)
        super().__init__(
            img_b64=base64.b64encode(png_bytes).decode(),
            max_idx=max_idx,
            init_idx=used_idx,
            init_axis=axis,
            px_mm_x=px_x,
            px_mm_y=px_y,
            **kwargs,
        )
        self.observe(self._on_state_change, names=["state"])

    def _on_state_change(self, change):
        try:
            s = json.loads(change["new"])
        except Exception:
            return

        axis  = s.get("axis", "axial")
        idx   = s.get("slice", None)
        tool  = s.get("tool", "distance")
        pts   = s.get("points", [])

        # Re-render slice PNG when axis/slice changes (no points yet)
        png_bytes, max_idx, _ = _render_png(
            self._nifti_path, axis, idx
        )
        self.img_b64  = base64.b64encode(png_bytes).decode()
        self.max_idx  = max_idx

        # Update pixdim for the new axis
        px_x, px_y = _get_pixdim(self._nifti_path, axis)
        self.px_mm_x = px_x
        self.px_mm_y = px_y

        # Intensity probe: resolve raw voxel value server-side
        # JS sends the completed point under 'intensity_probe' so it
        # arrives even after the JS POINTS array has been cleared.
        probe = s.get("intensity_probe")
        if tool == "intensity" and probe:
            slice_idx = idx if idx is not None else max_idx // 2
            val = _get_voxel_value(
                self._nifti_path, axis, slice_idx,
                probe["x"], probe["y"]
            )
            self.results = json.dumps([{"x": probe["x"], "y": probe["y"], "value": val}])

    @property
    def last_results(self) -> list[dict]:
        try:
            return json.loads(self.results)
        except Exception:
            return []