from ipyniivue import NiiVue, SliceType, ShowRender, Volume, ColormapType


def _set_slice_type(nv, slice_type):
    if slice_type is None:
        slice_type = SliceType.MULTIPLANAR
    nv.set_slice_type(slice_type)
    if slice_type == SliceType.RENDER:
        # Some widget frontends miss enum updates after volume loading; Niivue's
        # RENDER slice type is integer 4, so this makes the switch explicit.
        nv.set_slice_type(4)


def create_viewer(nifti_path, overlay_volumes=None, slice_type=None, show_render=False):
    if not nifti_path:
        return None

    if slice_type is None:
        slice_type = SliceType.RENDER if show_render else SliceType.MULTIPLANAR
    render_mode = ShowRender.ALWAYS if show_render else ShowRender.NEVER
    nv = NiiVue(
        show_3d_crosshair=True,
        multiplanar_show_render=render_mode,
        slice_type=slice_type,
        hero_slice_type=SliceType.RENDER if show_render else SliceType.SAGITTAL,
        hero_image_fraction=1.0 if show_render else 0.0,
        is_colorbar=True,
        is_nearest_interpolation=True,
    )

    all_volumes = [
        Volume(path=nifti_path, colormap="gray", opacity=1.0)
    ]

    if overlay_volumes:
        for ov in overlay_volumes:
            opacity = float(ov.get("opacity", 0.5))
            # Clamp opacity to a sensible range so overlays are always visible
            opacity = max(0.1, min(1.0, opacity))

            # ZERO_TO_MAX_TRANSPARENT_BELOW_MIN makes zero-valued background
            # voxels fully transparent, so only the labelled/segmented regions
            # appear as coloured overlays on top of the anatomical scan.
            colormap_type = ColormapType.ZERO_TO_MAX_TRANSPARENT_BELOW_MIN

            # For label/atlas overlays (integer index maps) the label colormap
            # type should stay as-is; for continuous maps use MIN_TO_MAX.
            # The caller can override by passing a "colormap_type" key.
            if "colormap_type" in ov:
                colormap_type = ov["colormap_type"]

            kwargs = {
                "path": ov["path"],
                "colormap": ov.get("colormap", "warm"),
                "opacity": opacity,
                "colormap_type": colormap_type,
            }
            if "cal_min" in ov:
                kwargs["cal_min"] = float(ov["cal_min"])
            if "cal_max" in ov:
                kwargs["cal_max"] = float(ov["cal_max"])
            all_volumes.append(Volume(**kwargs))

    _set_slice_type(nv, slice_type)
    nv.load_volumes(all_volumes)
    _set_slice_type(nv, slice_type)

    return nv
