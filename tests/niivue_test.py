from ipyniivue import NiiVue, ShowRender

# viewer
nv = NiiVue(
    back_color=(0, 0, 0, 1),
    show_3d_crosshair=True,
    multiplanar_show_render=ShowRender.ALWAYS,
)

# load pine MRI
nv.load_volumes([
    {
        "path": "data/data-multi-subject/sub-brnoCeitec01/anat/sub-brnoCeitec01_T2w.nii.gz"
    }
])

# adjust view (its optional)
nv.set_clip_plane(-0.2, 0, 120)

nv