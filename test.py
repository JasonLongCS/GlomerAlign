import json
from pathlib import Path

import napari
import numpy as np
from magicgui import magicgui
from skimage.io import imread


viewer = napari.Viewer(ndisplay=3)

fixed_layer = None
moving_layer = None
current_fixed = None
current_moving = None

def prepare_volume(img):
    img = np.asarray(img)
    img = np.squeeze(img)

    # RGB/RGBA 2D image: convert to grayscale
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img[..., :3].mean(axis=-1)

    # 4D stack with channel last: z, y, x, c
    if img.ndim == 4 and img.shape[-1] in (3, 4):
        img = img[..., :3].mean(axis=-1)

    # 4D stack with channel second: z, c, y, x
    elif img.ndim == 4:
        # choose first channel
        img = img[:, 0, :, :]

    if img.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {img.shape}")

    return img

def rotation_matrix_3d(rx_deg, ry_deg, rz_deg):
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)

    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx),  np.cos(rx), 0],
        [0, 0, 0, 1],
    ])

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1],
    ])

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz),  np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    return Rz @ Ry @ Rx


def make_affine_3d(rx, ry, rz, tz, ty, tx, center_zyx):
    cz, cy, cx = center_zyx

    T1 = np.eye(4)
    T1[:3, 3] = [-cz, -cy, -cx]

    T2 = np.eye(4)
    T2[:3, 3] = [cz, cy, cx]

    T3 = np.eye(4)
    T3[:3, 3] = [tz, ty, tx]

    R = rotation_matrix_3d(rx, ry, rz)

    return T3 @ T2 @ R @ T1


def update_transform():
    if moving_layer is None or current_moving is None:
        return

    rx = transform_widget.rx.value
    ry = transform_widget.ry.value
    rz = transform_widget.rz.value

    tz = transform_widget.tz.value
    ty = transform_widget.ty.value
    tx = transform_widget.tx.value

    z_thickness = transform_widget.z_thickness.value

    center = (np.array(current_moving.shape) - 1) / 2
    sx = stretch_widget.sx.value
    sy = stretch_widget.sy.value
    sz = stretch_widget.sz.value

    moving_layer.scale = (sz, sy, sx)
    moving_layer.affine = make_affine_3d(
        rx, ry, rz,
        tz, ty, tx,
        center
    )


@magicgui(
    fixed_path={"mode": "r"},
    moving_path={"mode": "r"},
    call_button="Load Images"
)
def load_images(
    fixed_path: Path = Path(),
    moving_path: Path = Path()
):
    global fixed_layer, moving_layer
    global current_fixed, current_moving

    if not fixed_path.exists() or not moving_path.exists():
        return

    fixed_vol = prepare_volume(imread(str(fixed_path)))
    moving_vol = prepare_volume(imread(str(moving_path)))

    current_fixed = fixed_vol
    current_moving = moving_vol

    viewer.layers.clear()

    fixed_layer = viewer.add_image(
        current_fixed,
        name="fixed",
        colormap="gray",
        opacity=1.0,
        scale=(transform_widget.z_thickness.value, 1, 1),
        rendering="mip",
    )

    moving_layer = viewer.add_image(
        current_moving,
        name="moving",
        colormap="magenta",
        opacity=0.5,
        scale=(transform_widget.z_thickness.value, 1, 1),
        rendering="mip",
    )

    update_transform()

@magicgui(
    sx={"widget_type": "FloatSlider", "min": 0.1, "max": 5},
    sy={"widget_type": "FloatSlider", "min": 0.1, "max": 5},
    sz={"widget_type": "FloatSlider", "min": 0.1, "max": 100},
)
def stretch_widget(
    sx: float = 1,
    sy: float = 1,
    sz: float = 20,
):
    update_transform()

@magicgui(
    rx={"widget_type": "FloatSlider", "min": -180, "max": 180},
    ry={"widget_type": "FloatSlider", "min": -180, "max": 180},
    rz={"widget_type": "FloatSlider", "min": -180, "max": 180},
    tz={"widget_type": "FloatSlider", "min": -500, "max": 500},
    ty={"widget_type": "FloatSlider", "min": -1000, "max": 1000},
    tx={"widget_type": "FloatSlider", "min": -1000, "max": 1000},
    z_thickness={"widget_type": "FloatSlider", "min": 0.1, "max": 100},
)
def transform_widget(
    rx: float = 0,
    ry: float = 0,
    rz: float = 0,
    tz: float = 0,
    ty: float = 0,
    tx: float = 0,
    z_thickness: float = 20,
):
    update_transform()




for w in [
    transform_widget.rx,
    transform_widget.ry,
    transform_widget.rz,
    transform_widget.tz,
    transform_widget.ty,
    transform_widget.tx,
    transform_widget.z_thickness,
]:
    w.changed.connect(lambda e: update_transform())


@magicgui(call_button="Save Transform")
def save_transform():
    if moving_layer is None:
        return

    data = {
        "rx": transform_widget.rx.value,
        "ry": transform_widget.ry.value,
        "rz": transform_widget.rz.value,
        "tz": transform_widget.tz.value,
        "ty": transform_widget.ty.value,
        "tx": transform_widget.tx.value,
        "z_thickness": transform_widget.z_thickness.value,
        "affine_matrix": np.asarray(moving_layer.affine.affine_matrix).tolist(),
        "scale": list(moving_layer.scale),
    }

    with open("alignment_transform.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Saved alignment_transform.json")


@magicgui(call_button="Load Transform")
def load_transform():
    with open("alignment_transform.json", "r") as f:
        data = json.load(f)

    transform_widget.rx.value = data["rx"]
    transform_widget.ry.value = data["ry"]
    transform_widget.rz.value = data["rz"]
    transform_widget.tz.value = data["tz"]
    transform_widget.ty.value = data["ty"]
    transform_widget.tx.value = data["tx"]
    transform_widget.z_thickness.value = data["z_thickness"]

    update_transform()


viewer.window.add_dock_widget(load_images, area="right")
viewer.window.add_dock_widget(stretch_widget, area="right")
viewer.window.add_dock_widget(transform_widget, area="right")
viewer.window.add_dock_widget(save_transform, area="right")
viewer.window.add_dock_widget(load_transform, area="right")

napari.run()