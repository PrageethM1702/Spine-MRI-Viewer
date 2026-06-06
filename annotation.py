"""
annotation.py — Feature 9: Annotation + Screenshot

Supports:
  • Arrow annotations (click start → click end)
  • Text label annotations (type text → click to place)

State is synced via the `state` traitlet as JSON:
  {
    "axis": str,
    "slice": int,
    "arrows": [{"x0","y0","x1","y1"}, ...],
    "labels": [{"x","y","text"}, ...]
  }
"""
import io
import json
import base64
import anywidget
import traitlets
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Slice / PNG helpers ───────────────────────────────────────────────────────

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


def render_slice_png(
    nifti_path: str,
    axis: str = "axial",
    slice_index: int | None = None,
    arrows: list[dict] | None = None,
    labels: list[dict] | None = None,
    fig_size: int = 500,
) -> tuple[bytes, int, int]:
    slice_data, max_idx, used_idx = _get_slice(nifti_path, axis, slice_index)
    dpi = 100
    size_in = fig_size / dpi
    fig, ax = plt.subplots(figsize=(size_in, size_in), dpi=dpi)
    ax.imshow(slice_data, cmap="gray", origin="upper", aspect="equal")
    ax.axis("off")

    h, w = slice_data.shape

    # Draw arrows
    if arrows:
        for a in arrows:
            ax.annotate(
                "",
                xy=(a["x1"] * w, a["y1"] * h),
                xytext=(a["x0"] * w, a["y0"] * h),
                arrowprops=dict(
                    arrowstyle="->,head_width=0.4,head_length=0.3",
                    color="#FF3B30",
                    lw=2.5,
                ),
            )

    # Draw text labels
    if labels:
        for lbl in labels:
            ax.text(
                lbl["x"] * w,
                lbl["y"] * h,
                lbl["text"],
                color="#FFD600",
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="left",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    alpha=0.55,
                    edgecolor="none",
                ),
            )

    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue(), max_idx, used_idx


# ── anywidget ─────────────────────────────────────────────────────────────────

class AnnotationWidget(anywidget.AnyWidget):
    """
    Annotation canvas widget supporting arrows and text labels.

    Python-writable traitlets:
        img_b64   — base64 PNG of the current slice (no annotations)
        max_idx   — maximum slice index
        init_idx  — initial slice index
        init_axis — initial axis string

    JS-writable traitlets (synced back to Python):
        state     — JSON: {"axis", "slice", "arrows": [...], "labels": [...]}

    Read-only convenience property:
        png_bytes — raw PNG bytes with annotations baked in
    """

    # Python → JS
    img_b64   = traitlets.Unicode("").tag(sync=True)
    max_idx   = traitlets.Int(0).tag(sync=True)
    init_idx  = traitlets.Int(0).tag(sync=True)
    init_axis = traitlets.Unicode("axial").tag(sync=True)

    # JS → Python
    state = traitlets.Unicode("{}").tag(sync=True)

    # Internal
    _nifti_path: str = ""
    _png_b64: str = ""

    _esm = r"""
function render({ model, el }) {
  const FIG = 500;

  // ── DOM ───────────────────────────────────────────────────────────────────
  el.style.fontFamily = "sans-serif";
  el.style.userSelect = "none";

  // Controls row 1: axis + slice
  const row1 = document.createElement("div");
  row1.style.cssText = "display:flex;gap:10px;align-items:center;margin-bottom:8px;flex-wrap:wrap";

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
  const sliceRange = document.createElement("input");
  sliceRange.type = "range";
  sliceRange.min = 0;
  sliceRange.style.width = "140px";
  const maxSpan = document.createElement("span");
  sliceLabel.append("Slice: ", sliceNum, " / ");
  sliceLabel.appendChild(maxSpan);

  row1.append(axisLabel, axisSelect, sliceLabel, sliceRange);

  // Controls row 2: mode + label input + action buttons
  const row2 = document.createElement("div");
  row2.style.cssText = "display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap";

  // Mode toggle
  const modeArrowBtn = makeBtn("✏️ Arrow", () => setMode("arrow"));
  const modeLabelBtn = makeBtn("🔤 Text Label", () => setMode("label"));
  modeArrowBtn.style.background = "#FF3B30";
  modeArrowBtn.style.color = "#fff";
  modeArrowBtn.style.border = "none";

  // Text input (shown only in label mode)
  const labelInput = document.createElement("input");
  labelInput.type = "text";
  labelInput.placeholder = "Type label text…";
  labelInput.style.cssText = "font-size:13px;padding:4px 8px;border-radius:4px;border:1px solid #ccc;width:160px;display:none";

  const undoBtn = makeBtn("↩ Undo", undoLast);
  const clearBtn = makeBtn("🗑 Clear All", clearAll);

  row2.append(modeArrowBtn, modeLabelBtn, labelInput, undoBtn, clearBtn);

  // Canvas wrapper
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "position:relative;display:inline-block;border:1px solid #ddd;border-radius:4px;overflow:hidden;line-height:0";

  const img = document.createElement("img");
  img.style.cssText = `display:block;width:${FIG}px;height:${FIG}px`;
  img.draggable = false;

  const canvas = document.createElement("canvas");
  canvas.width = FIG;
  canvas.height = FIG;
  canvas.style.cssText = `position:absolute;top:0;left:0;width:${FIG}px;height:${FIG}px;cursor:crosshair`;

  wrapper.append(img, canvas);

  const hint = document.createElement("div");
  hint.style.cssText = "margin-top:8px;font-size:12px;color:#888";
  hint.textContent = "Arrow mode: click start then end. Text mode: type label then click to place.";

  el.append(row1, row2, wrapper, hint);

  // ── State ─────────────────────────────────────────────────────────────────
  const ctx = canvas.getContext("2d");
  let MODE    = "arrow";   // "arrow" | "label"
  let ARROWS  = [];
  let LABELS  = [];
  let PENDING = null;      // for arrow: {x,y} start point

  function makeBtn(label, handler) {
    const b = document.createElement("button");
    b.textContent = label;
    b.style.cssText = "padding:4px 10px;font-size:13px;border-radius:4px;border:1px solid #ccc;cursor:pointer;background:#fff";
    b.addEventListener("click", handler);
    return b;
  }

  function setMode(m) {
    MODE = m;
    PENDING = null;
    if (m === "arrow") {
      modeArrowBtn.style.background = "#FF3B30";
      modeArrowBtn.style.color = "#fff";
      modeArrowBtn.style.border = "none";
      modeLabelBtn.style.background = "#fff";
      modeLabelBtn.style.color = "#000";
      modeLabelBtn.style.border = "1px solid #ccc";
      labelInput.style.display = "none";
      canvas.style.cursor = "crosshair";
    } else {
      modeLabelBtn.style.background = "#FFD600";
      modeLabelBtn.style.color = "#000";
      modeLabelBtn.style.border = "none";
      modeArrowBtn.style.background = "#fff";
      modeArrowBtn.style.color = "#000";
      modeArrowBtn.style.border = "1px solid #ccc";
      labelInput.style.display = "inline-block";
      labelInput.focus();
      canvas.style.cursor = "text";
    }
    redraw();
  }

  // ── Init ──────────────────────────────────────────────────────────────────
  function initFromModel() {
    const maxI = model.get("max_idx");
    const initI = model.get("init_idx");
    const initA = model.get("init_axis");
    sliceRange.max = maxI;
    sliceRange.value = initI;
    maxSpan.textContent = maxI;
    sliceNum.textContent = initI;
    axisSelect.value = initA;
    updateImage();
  }

  function updateImage() {
    const b64 = model.get("img_b64");
    if (b64) img.src = "data:image/png;base64," + b64;
  }

  // ── Push state to Python ──────────────────────────────────────────────────
  function pushState() {
    model.set("state", JSON.stringify({
      axis:   axisSelect.value,
      slice:  parseInt(sliceRange.value),
      arrows: ARROWS,
      labels: LABELS,
    }));
    model.save_changes();
  }

  // ── Canvas drawing ────────────────────────────────────────────────────────
  function redraw() {
    ctx.clearRect(0, 0, FIG, FIG);
    ARROWS.forEach(a => drawArrow(a.x0*FIG, a.y0*FIG, a.x1*FIG, a.y1*FIG));
    LABELS.forEach(l => drawLabel(l.x*FIG, l.y*FIG, l.text));
    if (PENDING) drawDot(PENDING.x*FIG, PENDING.y*FIG);
  }

  function drawArrow(x0, y0, x1, y1) {
    const dx = x1-x0, dy = y1-y0;
    if (Math.sqrt(dx*dx+dy*dy) < 4) return;
    ctx.strokeStyle = "#FF3B30";
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(x0,y0); ctx.lineTo(x1,y1); ctx.stroke();
    const angle = Math.atan2(dy,dx), hs = 12;
    ctx.fillStyle = "#FF3B30";
    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x1-hs*Math.cos(angle-0.4), y1-hs*Math.sin(angle-0.4));
    ctx.lineTo(x1-hs*Math.cos(angle+0.4), y1-hs*Math.sin(angle+0.4));
    ctx.closePath(); ctx.fill();
  }

  function drawLabel(x, y, text) {
    if (!text) return;
    ctx.font = "bold 14px sans-serif";
    const metrics = ctx.measureText(text);
    const pad = 4;
    const bw = metrics.width + pad*2;
    const bh = 18 + pad;
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.beginPath();
    ctx.roundRect(x, y, bw, bh, 3);
    ctx.fill();
    ctx.fillStyle = "#FFD600";
    ctx.fillText(text, x + pad, y + bh - pad - 2);
  }

  function drawDot(x, y) {
    ctx.fillStyle = "#FF3B30";
    ctx.beginPath(); ctx.arc(x,y,5,0,2*Math.PI); ctx.fill();
  }

  // ── Canvas click ──────────────────────────────────────────────────────────
  canvas.addEventListener("click", e => {
    const r = canvas.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width;
    const y = (e.clientY - r.top)  / r.height;

    if (MODE === "label") {
      const text = labelInput.value.trim();
      if (!text) {
        labelInput.focus();
        labelInput.style.border = "1px solid #FF3B30";
        setTimeout(() => labelInput.style.border = "1px solid #ccc", 800);
        return;
      }
      LABELS.push({x, y, text});
      redraw();
      pushState();
      // keep text for repeated labels, user can clear manually
    } else {
      // arrow mode
      if (PENDING === null) {
        PENDING = {x, y};
      } else {
        ARROWS.push({x0: PENDING.x, y0: PENDING.y, x1: x, y1: y});
        PENDING = null;
        pushState();
      }
      redraw();
    }
  });

  axisSelect.addEventListener("change", () => { PENDING = null; ctx.clearRect(0,0,FIG,FIG); pushState(); });
  sliceRange.addEventListener("input", () => {
    sliceNum.textContent = sliceRange.value;
    PENDING = null; ctx.clearRect(0,0,FIG,FIG); pushState();
  });

  function undoLast() {
    if (PENDING !== null) { PENDING = null; redraw(); return; }
    if (MODE === "label" && LABELS.length > 0) { LABELS.pop(); redraw(); pushState(); return; }
    if (ARROWS.length > 0) { ARROWS.pop(); redraw(); pushState(); }
  }

  function clearAll() {
    ARROWS = []; LABELS = []; PENDING = null; redraw(); pushState();
  }

  // ── Model listeners ───────────────────────────────────────────────────────
  model.on("change:img_b64", updateImage);
  model.on("change:max_idx", () => {
    sliceRange.max = model.get("max_idx");
    maxSpan.textContent = model.get("max_idx");
  });

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
        png_bytes, max_idx, used_idx = render_slice_png(
            nifti_path, axis, slice_index, arrows=None, labels=None
        )
        super().__init__(
            img_b64=base64.b64encode(png_bytes).decode(),
            max_idx=max_idx,
            init_idx=used_idx,
            init_axis=axis,
            **kwargs,
        )
        self._png_b64 = base64.b64encode(png_bytes).decode()
        self.observe(self._on_state_change, names=["state"])

    def _on_state_change(self, change):
        try:
            s = json.loads(change["new"])
        except Exception:
            return
        axis   = s.get("axis", "axial")
        idx    = s.get("slice", None)
        arrows = s.get("arrows", [])
        labels = s.get("labels", [])

        png_bytes, max_idx, _ = render_slice_png(
            self._nifti_path, axis, idx, arrows, labels
        )
        self._png_b64 = base64.b64encode(png_bytes).decode()
        self.img_b64  = self._png_b64
        self.max_idx  = max_idx

    @property
    def png_bytes(self) -> bytes:
        return base64.b64decode(self._png_b64)