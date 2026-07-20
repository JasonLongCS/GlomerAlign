import sys
import os
import json
from pathlib import Path

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yaml
from tifffile import imread, imwrite
from skimage.measure import regionprops_table
from skimage.transform import AffineTransform, PiecewiseAffineTransform, SimilarityTransform, warp

import napari
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
    QCheckBox,
    QInputDialog,
    QTabWidget,
)

try:
    from magicgui import magicgui  # noqa: F401
    MAGICGUI_AVAILABLE = True
except Exception:
    MAGICGUI_AVAILABLE = False


# -----------------------------
# Global paths/config
# -----------------------------
MATCHES_DIR = "matches"
os.makedirs(MATCHES_DIR, exist_ok=True)

CONFIG = {}
TRANSFORM_CONFIG = {}
TRANSFORM_PATH = None

INVIVO_IMAGE = "In-vivo Image"
INVIVO_MASK = "In-vivo Mask"
INVIVO_MATCHES = "In-vivo Matches"

INVITRO_IMAGE = "In-vitro Image"
INVITRO_MASK = "In-vitro Mask"
INVITRO_MATCHES = "In-vitro Matches"

DEFORM_SRC_POINTS = "Deform Source Points"
DEFORM_DST_POINTS = "Deform Destination Points"
DEFORM_GRID = "Deform Grid"
TRAKEM_SRC_POINTS = "Nonlinear Source Points"
TRAKEM_DST_POINTS = "Nonlinear Destination Points"
DEFORMATION_PATH = None
MAX_DEFORMATION_GRID_POINTS = 2500


# -----------------------------
# Config helpers
# -----------------------------
def _get_scale(ndim):
    """Return scale tuple from config. 3D uses (z, y, x), 2D uses (y, x)."""
    scale_cfg = CONFIG.get("scale", {}) if CONFIG else {}
    xy = float(scale_cfg.get("xy", 1.0))
    z = float(scale_cfg.get("z", 1.0))
    if ndim == 2:
        return (xy, xy)
    if ndim == 3:
        return (z, xy, xy)
    return tuple([1.0] * ndim)


def load_global_config(config_path=None):
    global CONFIG
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            CONFIG = yaml.safe_load(f) or {}
        print(f"Config loaded from {config_path}")
        return True
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        CONFIG = {"models": {}}
        return False


def load_transform_config(config_path=None):
    global TRANSFORM_CONFIG, TRANSFORM_PATH
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "alignment_transform.json")
    TRANSFORM_PATH = config_path

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            TRANSFORM_CONFIG = json.load(f)
        print(f"Transform config loaded from {config_path}")
        return True
    except FileNotFoundError:
        print(f"Transform config not found: {config_path}")
        TRANSFORM_CONFIG = {}
        return False


# -----------------------------
# Affine helpers
# -----------------------------
def rotation_matrix_3d(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1],
    ], dtype=float)

    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1],
    ], dtype=float)

    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)

    return Rz @ Ry @ Rx


def make_affine_3d(rx, ry, rz, tz, ty, tx, sz, sy, sx, center_zyx):
    """Make a 4x4 affine for napari 3D data in z, y, x order."""
    cz, cy, cx = center_zyx

    to_origin = np.eye(4)
    to_origin[:3, 3] = [-cz, -cy, -cx]

    back = np.eye(4)
    back[:3, 3] = [cz, cy, cx]

    translate = np.eye(4)
    translate[:3, 3] = [tz, ty, tx]

    scale = np.eye(4)
    scale[0, 0] = sz
    scale[1, 1] = sy
    scale[2, 2] = sx

    rotate = rotation_matrix_3d(rx, ry, rz)
    return translate @ back @ rotate @ scale @ to_origin


def add_or_replace_image(viewer, data, name, opacity=1.0):
    scale = _get_scale(data.ndim)
    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        layer.scale = scale
        layer.opacity = opacity
        layer.refresh()
    else:
        viewer.add_image(data, name=name, opacity=opacity, scale=scale)


def add_or_replace_labels(viewer, data, name, opacity=0.35):
    scale = _get_scale(data.ndim)
    if name in viewer.layers:
        layer = viewer.layers[name]
        layer.data = data
        layer.scale = scale
        layer.opacity = opacity
        layer.refresh()
    else:
        viewer.add_labels(data, name=name, opacity=opacity, scale=scale)


def read_tiff(path):
    """Read TIFF. Kept as a single function so you can swap to memmap later if needed."""
    return imread(path)




# -----------------------------
# Slice deformation helpers
# -----------------------------
def current_z_index(viewer):
    """Return the current z index for 3D data shown in 2D mode."""
    try:
        z = int(round(viewer.dims.current_step[0]))
    except Exception:
        z = 0
    return max(z, 0)


def get_layer_slice_2d(layer, z):
    data = layer.data
    if data.ndim == 2:
        return data
    z = min(max(int(z), 0), data.shape[0] - 1)
    return data[z]


def set_layer_slice_2d(layer, z, slice_data):
    """Assign a warped 2D slice back into a 2D/3D napari layer.

    Important: after modifying a slice in-place, reassign layer.data to the
    same array so napari emits a data-change event. layer.refresh() alone can
    fail to visibly redraw for large arrays / cached rendering.
    """
    data = layer.data
    if data.ndim == 2:
        layer.data = slice_data.astype(data.dtype, copy=False)
    else:
        z = min(max(int(z), 0), data.shape[0] - 1)
        data[z, ...] = slice_data.astype(data.dtype, copy=False)
        layer.data = data
    layer.refresh()


def get_invitro_2d_scale(viewer):
    """Return y/x scale for deformation points so they overlay image data."""
    if INVITRO_IMAGE in viewer.layers:
        scale = viewer.layers[INVITRO_IMAGE].scale
    elif INVITRO_MASK in viewer.layers:
        scale = viewer.layers[INVITRO_MASK].scale
    else:
        return (1.0, 1.0)
    if len(scale) >= 2:
        return tuple(scale[-2:])
    return (1.0, 1.0)


def deformation_grid_shape(shape_yx, spacing):
    """Return regular grid row/column counts without allocating its points."""
    if len(shape_yx) != 2:
        raise ValueError("A deformation grid requires a 2D (height, width) shape.")
    height, width = (int(value) for value in shape_yx)
    if height < 2 or width < 2:
        raise ValueError("A deformation grid requires an image at least 2 x 2 pixels.")
    spacing = max(int(spacing), 2)
    ny = ((height - 2) // spacing) + 2
    nx = ((width - 2) // spacing) + 2
    return ny, nx


def make_regular_grid_points(shape_yx, spacing):
    """Return points in napari 2D coordinates, y/x order."""
    deformation_grid_shape(shape_yx, spacing)
    height, width = (int(value) for value in shape_yx)
    spacing = max(int(spacing), 2)

    ys = list(range(0, height, spacing))
    xs = list(range(0, width, spacing))
    if ys[-1] != height - 1:
        ys.append(height - 1)
    if xs[-1] != width - 1:
        xs.append(width - 1)

    points = []
    for y in ys:
        for x in xs:
            points.append([float(y), float(x)])
    return np.asarray(points, dtype=float), ys, xs


def minimum_grid_spacing(shape_yx, max_points=MAX_DEFORMATION_GRID_POINTS):
    """Return the smallest spacing whose regular grid stays under max_points."""
    if max_points < 4:
        raise ValueError("A deformation grid needs room for at least four points.")
    deformation_grid_shape(shape_yx, 2)
    spacing = 2
    while True:
        ny, nx = deformation_grid_shape(shape_yx, spacing)
        if ny * nx <= max_points:
            return spacing
        spacing += 1


def make_grid_lines_from_points(points_yx, ys, xs):
    """Create napari Shapes paths from grid points in y/x order."""
    point_lookup = {(int(round(y)), int(round(x))): np.array([y, x], dtype=float) for y, x in points_yx}
    lines = []

    for y in ys:
        row = []
        for x in xs:
            row.append(point_lookup[(int(y), int(x))])
        lines.append(np.asarray(row, dtype=float))

    for x in xs:
        col = []
        for y in ys:
            col.append(point_lookup[(int(y), int(x))])
        lines.append(np.asarray(col, dtype=float))

    return lines


def points_yx_to_xy(points_yx):
    """skimage transforms use x/y coordinates, while napari points are y/x."""
    points_yx = np.asarray(points_yx, dtype=float)
    return points_yx[:, [1, 0]]


def validate_deformation_points(src_yx, dst_yx):
    """Validate point arrays before handing them to scipy/Qhull via skimage."""
    src = np.asarray(src_yx, dtype=float)
    dst = np.asarray(dst_yx, dtype=float)
    if src.ndim != 2 or src.shape[1:] != (2,):
        raise RuntimeError("Deformation points must be an N x 2 array in y/x order.")
    if src.shape != dst.shape:
        raise RuntimeError("Source and destination points must have the same shape.")
    if len(src) < 4:
        raise RuntimeError("A deformation grid requires at least four control points.")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise RuntimeError("Deformation points contain NaN or infinite coordinates.")
    for name, points in (("Source", src), ("Destination", dst)):
        if len(np.unique(points, axis=0)) < 3:
            raise RuntimeError(f"{name} points contain too few unique positions.")
        if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise RuntimeError(f"{name} points are collinear and cannot define a 2D warp.")
    return src, dst


def warp_slice_with_points(slice_2d, src_yx, dst_yx, order):
    """
    Warp slice using manually moved control points.

    src_yx = original control grid positions.
    dst_yx = edited/moved control grid positions.

    The transform maps src -> dst. skimage.warp needs inverse_map, so this
    samples the original source image into the destination image.
    """
    src_yx, dst_yx = validate_deformation_points(src_yx, dst_yx)
    src_xy = points_yx_to_xy(src_yx)
    dst_xy = points_yx_to_xy(dst_yx)

    try:
        tform = PiecewiseAffineTransform()
        ok = tform.estimate(src_xy, dst_xy)
    except Exception as exc:
        raise RuntimeError(
            "Could not triangulate the deformation grid. Move overlapping or "
            "crossed destination points closer to their source positions."
        ) from exc
    if not ok:
        raise RuntimeError("Could not estimate piecewise affine transform. Try more grid points or less extreme deformation.")

    warped = warp(
        slice_2d,
        inverse_map=tform.inverse,
        output_shape=slice_2d.shape,
        order=order,
        preserve_range=True,
        mode="edge",
    )
    return warped


def empty_points():
    """Return an empty napari-compatible 2D point array."""
    return np.empty((0, 2), dtype=float)


def nonlinear_transform_for_points(src_yx, dst_yx, output_shape=None):
    """Estimate model by landmark count: translation, similarity, affine, or local."""
    src, dst = np.asarray(src_yx, float), np.asarray(dst_yx, float)
    if src.ndim != 2 or src.shape[1:] != (2,) or src.shape != dst.shape:
        raise RuntimeError("Control points must be matching N x 2 arrays in y/x order.")
    if len(src) < 1:
        raise RuntimeError("Add at least one nonlinear control point.")
    if not np.all(np.isfinite(src)) or not np.all(np.isfinite(dst)):
        raise RuntimeError("Control points contain NaN or infinite coordinates.")
    src_xy, dst_xy = points_yx_to_xy(src), points_yx_to_xy(dst)
    if len(src) == 1:
        return SimilarityTransform(translation=dst_xy[0] - src_xy[0]), "translation"
    if len(src) == 2:
        if np.linalg.norm(src_xy[1] - src_xy[0]) < 1e-6:
            raise RuntimeError("The two source points must be distinct.")
        tform, model = SimilarityTransform(), "similarity"
        ok = tform.estimate(src_xy, dst_xy)
    elif len(src) == 3:
        if np.linalg.matrix_rank(src - src.mean(axis=0)) < 2:
            raise RuntimeError("Three points must not be collinear.")
        tform, model = AffineTransform(), "affine"
        ok = tform.estimate(src_xy, dst_xy)
    else:
        if output_shape is None:
            raise RuntimeError("An output shape is required for a local transform.")
        height, width = output_shape
        boundary = np.asarray([[0, 0], [width-1, 0], [0, height-1], [width-1, height-1],
            [(width-1)/2, 0], [(width-1)/2, height-1],
            [0, (height-1)/2], [width-1, (height-1)/2]], float)
        keep = np.min(np.linalg.norm(boundary[:, None] - src_xy[None], axis=2), axis=1) > 1e-6
        boundary = boundary[keep]
        src_xy, dst_xy = np.vstack([src_xy, boundary]), np.vstack([dst_xy, boundary])
        tform, model = PiecewiseAffineTransform(), "piecewise affine"
        try:
            ok = tform.estimate(src_xy, dst_xy)
        except Exception as exc:
            raise RuntimeError("Could not triangulate the local transform; check for duplicate points.") from exc
    if not ok:
        raise RuntimeError(f"Could not estimate the {model} transform.")
    return tform, model


def warp_slice_trakem2(slice_2d, src_yx, dst_yx, order):
    """Warp one slice using TrakEM2 landmark-count behavior."""
    tform, _ = nonlinear_transform_for_points(src_yx, dst_yx, slice_2d.shape)
    return warp(slice_2d, inverse_map=tform.inverse, output_shape=slice_2d.shape,
                order=order, preserve_range=True, mode="edge")


# -----------------------------
# Match handling in one viewer
# -----------------------------
class MatchHandler:
    def __init__(self, viewer):
        self.viewer = viewer
        self.clicked = {"in_vivo": None, "ex_vivo": None}
        self.glomeruli_path = os.path.join(MATCHES_DIR, "glomeruli.csv")
        self.undo_stack = []
        self.setup()

    def setup(self):
        self.viewer.bind_key("h", self.on_key_press, overwrite=True)
        self.viewer.bind_key("z", self.undo_match, overwrite=True)

    def on_key_press(self, viewer):
        active_layer = viewer.layers.selection.active
        if active_layer is None:
            print("Please select either In-vivo Mask or In-vitro Mask.")
            return

        if active_layer.name == INVIVO_MASK:
            viewer_name = "in_vivo"
        elif active_layer.name == INVITRO_MASK:
            viewer_name = "ex_vivo"
        else:
            print("Please select either In-vivo Mask or In-vitro Mask before pressing h.")
            return

        pos = viewer.cursor.position
        scale = active_layer.scale
        cursor_pos = []
        for i in range(active_layer.data.ndim):
            cursor_pos.append(pos[i] / scale[i])
        cursor_pos = tuple(map(int, np.round(cursor_pos)))

        try:
            selected_label = active_layer.data[cursor_pos]
        except IndexError:
            print(f"Cursor position outside bounds for {active_layer.name}: {cursor_pos}")
            return

        if selected_label == 0:
            print("Background selected, label 0. Please select a valid label.")
            return

        self.on_label_selected(viewer_name, int(selected_label))

    def on_label_selected(self, viewer_name, label):
        self.clicked[viewer_name] = label
        print(f"Selected label {label} from {viewer_name}")

        other = "ex_vivo" if viewer_name == "in_vivo" else "in_vivo"
        if self.clicked[other] is not None:
            self.record_match()

    def record_match(self):
        invivo_label = self.clicked["in_vivo"]
        invitro_label = self.clicked["ex_vivo"]
        if invivo_label is None or invitro_label is None:
            print("Need both an in-vivo and in-vitro label selected.")
            return

        required = [INVIVO_MASK, INVITRO_MASK, INVIVO_MATCHES, INVITRO_MATCHES]
        missing = [name for name in required if name not in self.viewer.layers]
        if missing:
            print(f"Missing layers: {missing}. Load masks and initialize matches first.")
            return

        invivo_seg = self.viewer.layers[INVIVO_MASK].data
        invitro_seg = self.viewer.layers[INVITRO_MASK].data
        invivo_match = self.viewer.layers[INVIVO_MATCHES].data
        invitro_match = self.viewer.layers[INVITRO_MATCHES].data

        color = invivo_label
        invivo_match[invivo_seg == invivo_label] = color
        invitro_match[invitro_seg == invitro_label] = color

        self.viewer.layers[INVIVO_MATCHES].refresh()
        self.viewer.layers[INVITRO_MATCHES].refresh()

        if os.path.exists(self.glomeruli_path):
            df = pd.read_csv(self.glomeruli_path, encoding="utf-8")
        else:
            df = pd.DataFrame(columns=["invivo", "exvivo", "color"])

        df.loc[len(df)] = [invivo_label, invitro_label, color]
        df.to_csv(self.glomeruli_path, index=False, encoding="utf-8")

        self.undo_stack.append((invivo_label, invitro_label, color))
        self.clicked = {"in_vivo": None, "ex_vivo": None}

        imwrite(os.path.join(MATCHES_DIR, "invivo_matches.tif"), invivo_match)
        imwrite(os.path.join(MATCHES_DIR, "invitro_matches.tif"), invitro_match)
        print(f"Matched in-vivo {invivo_label} to in-vitro {invitro_label}")

    def undo_match(self, viewer):
        if not self.undo_stack:
            print("No matches to undo.")
            return

        invivo_label, invitro_label, color = self.undo_stack.pop()
        if INVIVO_MATCHES not in self.viewer.layers or INVITRO_MATCHES not in self.viewer.layers:
            print("Match layers not found.")
            return

        invivo_match = self.viewer.layers[INVIVO_MATCHES].data
        invitro_match = self.viewer.layers[INVITRO_MATCHES].data
        invivo_match[invivo_match == color] = 0
        invitro_match[invitro_match == color] = 0

        self.viewer.layers[INVIVO_MATCHES].refresh()
        self.viewer.layers[INVITRO_MATCHES].refresh()

        if os.path.exists(self.glomeruli_path):
            df = pd.read_csv(self.glomeruli_path, encoding="utf-8")
            df = df[~((df["invivo"] == invivo_label) & (df["exvivo"] == invitro_label) & (df["color"] == color))]
            df.to_csv(self.glomeruli_path, index=False, encoding="utf-8")

        imwrite(os.path.join(MATCHES_DIR, "invivo_matches.tif"), invivo_match)
        imwrite(os.path.join(MATCHES_DIR, "invitro_matches.tif"), invitro_match)
        print(f"Undid match between in-vivo {invivo_label} and in-vitro {invitro_label}")


class MatchLoader:
    def __init__(self, viewer):
        self.viewer = viewer

    def load_matches(self):
        os.makedirs(MATCHES_DIR, exist_ok=True)
        base_csv = os.path.join(MATCHES_DIR, "glomeruli.csv")
        invivo_matches_path = os.path.join(MATCHES_DIR, "invivo_matches.tif")
        invitro_matches_path = os.path.join(MATCHES_DIR, "invitro_matches.tif")

        if INVIVO_MASK not in self.viewer.layers or INVITRO_MASK not in self.viewer.layers:
            QMessageBox.warning(None, "Masks Required", "Load both in-vivo and in-vitro masks first.")
            return

        invivo_seg = self.viewer.layers[INVIVO_MASK].data
        invitro_seg = self.viewer.layers[INVITRO_MASK].data

        if os.path.exists(base_csv) and os.path.exists(invivo_matches_path) and os.path.exists(invitro_matches_path):
            print("Loading existing match layers.")
            invivo_data = read_tiff(invivo_matches_path)
            invitro_data = read_tiff(invitro_matches_path)
        else:
            print("Creating new match layers and CSVs.")
            invivo_data = np.zeros_like(invivo_seg, dtype=np.uint16)
            invitro_data = np.zeros_like(invitro_seg, dtype=np.uint16)

            self._get_region_table(invivo_seg).to_csv(
                os.path.join(MATCHES_DIR, "invivo_glomeruli.csv"), index=False, encoding="utf-8"
            )
            self._get_region_table(invitro_seg).to_csv(
                os.path.join(MATCHES_DIR, "invitro_glomeruli.csv"), index=False, encoding="utf-8"
            )
            pd.DataFrame(columns=["invivo", "exvivo", "color"]).to_csv(base_csv, index=False, encoding="utf-8")
            imwrite(invivo_matches_path, invivo_data)
            imwrite(invitro_matches_path, invitro_data)

        add_or_replace_labels(self.viewer, invivo_data, INVIVO_MATCHES, opacity=0.7)
        add_or_replace_labels(self.viewer, invitro_data, INVITRO_MATCHES, opacity=0.7)

    def _get_region_table(self, seg):
        props = regionprops_table(seg, properties=("label", "centroid"))
        df = pd.DataFrame(props)

        if seg.ndim == 2:
            df.columns = ["id", "y", "x"]
            df["z"] = 0
        elif seg.ndim == 3:
            df.columns = ["id", "z", "y", "x"]
        else:
            df["id"] = []

        df["color"] = df["id"]
        df["matched"] = False
        df["receptor"] = None
        return df


# -----------------------------
# Main dock widget
# -----------------------------
class ControlPanel(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.match_loader = MatchLoader(viewer)
        self.match_handler = MatchHandler(viewer)
        self.current_affine = np.eye(4)
        self.deformations = {}
        self.grid_spacing = 100
        self._nonlinear_mode = False
        self._building_ui = True

        layout = QVBoxLayout()

        # Compact right sidebar. Tabs avoid one very long vertical control panel,
        # but keep the same behavior as this version of the GUI.
        tabs = QTabWidget()

        # -------------------------
        # Data / view / matching tab
        # -------------------------
        data_tab = QWidget()
        data_layout = QVBoxLayout()

        load_group = QGroupBox("Load")
        load_layout = QVBoxLayout()

        row = QHBoxLayout()
        self.load_invivo_image_button = QPushButton("In-vivo Img")
        self.load_invivo_image_button.clicked.connect(lambda: self.load_image(INVIVO_IMAGE))
        row.addWidget(self.load_invivo_image_button)

        self.load_invivo_mask_button = QPushButton("In-vivo Mask")
        self.load_invivo_mask_button.clicked.connect(lambda: self.load_mask(INVIVO_MASK))
        row.addWidget(self.load_invivo_mask_button)
        load_layout.addLayout(row)

        row = QHBoxLayout()
        self.load_invitro_image_button = QPushButton("In-vitro Img")
        self.load_invitro_image_button.clicked.connect(lambda: self.load_image(INVITRO_IMAGE))
        row.addWidget(self.load_invitro_image_button)

        self.load_invitro_mask_button = QPushButton("In-vitro Mask")
        self.load_invitro_mask_button.clicked.connect(lambda: self.load_mask(INVITRO_MASK))
        row.addWidget(self.load_invitro_mask_button)
        load_layout.addLayout(row)

        row = QHBoxLayout()
        self.load_from_config_button = QPushButton("Cfg Images")
        self.load_from_config_button.clicked.connect(self.load_images_from_config)
        row.addWidget(self.load_from_config_button)

        self.load_masks_from_config_button = QPushButton("Cfg Masks")
        self.load_masks_from_config_button.clicked.connect(self.load_masks_from_config)
        row.addWidget(self.load_masks_from_config_button)
        load_layout.addLayout(row)

        load_group.setLayout(load_layout)
        data_layout.addWidget(load_group)

        view_group = QGroupBox("View")
        view_layout = QHBoxLayout()
        self.ndisplay_checkbox = QCheckBox("3D display")
        self.ndisplay_checkbox.setChecked(False)
        self.ndisplay_checkbox.stateChanged.connect(self.toggle_3d)
        view_layout.addWidget(self.ndisplay_checkbox)
        view_group.setLayout(view_layout)
        data_layout.addWidget(view_group)

        match_group = QGroupBox("Matching")
        match_layout = QVBoxLayout()
        row = QHBoxLayout()
        self.load_matches_button = QPushButton("Load/Init")
        self.load_matches_button.clicked.connect(self.match_loader.load_matches)
        row.addWidget(self.load_matches_button)

        self.save_matches_button = QPushButton("Save")
        self.save_matches_button.clicked.connect(self.save_matches)
        row.addWidget(self.save_matches_button)
        match_layout.addLayout(row)
        match_layout.addWidget(QLabel("Select mask layer. h = pick, z = undo."))
        match_group.setLayout(match_layout)
        data_layout.addWidget(match_group)

        save_group = QGroupBox("Save")
        save_layout = QVBoxLayout()
        row = QHBoxLayout()
        self.save_invivo_mask_button = QPushButton("In-vivo Mask")
        self.save_invivo_mask_button.clicked.connect(lambda: self.save_layer(INVIVO_MASK))
        row.addWidget(self.save_invivo_mask_button)

        self.save_invitro_mask_button = QPushButton("In-vitro Mask")
        self.save_invitro_mask_button.clicked.connect(lambda: self.save_layer(INVITRO_MASK))
        row.addWidget(self.save_invitro_mask_button)
        save_layout.addLayout(row)

        self.save_selected_button = QPushButton("Selected Layer")
        self.save_selected_button.clicked.connect(self.save_selected_layer)
        save_layout.addWidget(self.save_selected_button)
        save_group.setLayout(save_layout)
        data_layout.addWidget(save_group)

        data_layout.addStretch(1)
        data_tab.setLayout(data_layout)
        tabs.addTab(data_tab, "Data")

        # -------------------------
        # Transform tab
        # -------------------------
        transform_tab = QWidget()
        transform_layout = QVBoxLayout()

        self.tx_slider, self.tx_spin = self.make_slider_spin("Tx", -5000, 5000, 0, transform_layout)
        self.ty_slider, self.ty_spin = self.make_slider_spin("Ty", -5000, 5000, 0, transform_layout)
        self.tz_slider, self.tz_spin = self.make_slider_spin("Tz", -5000, 5000, 0, transform_layout)

        self.rx_slider, self.rx_spin = self.make_slider_spin("Rx", -180, 180, 0, transform_layout)
        self.ry_slider, self.ry_spin = self.make_slider_spin("Ry", -180, 180, 0, transform_layout)
        self.rz_slider, self.rz_spin = self.make_slider_spin("Rz", -180, 180, 0, transform_layout)

        scale_row = QHBoxLayout()
        self.sx_spin = self.make_double_spin_inline("Sx", 0.01, 20.0, 1.0, scale_row)
        self.sy_spin = self.make_double_spin_inline("Sy", 0.01, 20.0, 1.0, scale_row)
        self.sz_spin = self.make_double_spin_inline("Sz", 0.01, 20.0, 1.0, scale_row)
        transform_layout.addLayout(scale_row)

        row = QHBoxLayout()
        self.apply_saved_button = QPushButton("Apply Saved")
        self.apply_saved_button.clicked.connect(self.apply_saved_transform)
        row.addWidget(self.apply_saved_button)

        self.save_transform_button = QPushButton("Save")
        self.save_transform_button.clicked.connect(self.save_transform)
        row.addWidget(self.save_transform_button)
        transform_layout.addLayout(row)

        self.reset_transform_button = QPushButton("Reset Controls")
        self.reset_transform_button.clicked.connect(self.reset_transform_controls)
        transform_layout.addWidget(self.reset_transform_button)

        transform_layout.addStretch(1)
        transform_tab.setLayout(transform_layout)
        tabs.addTab(transform_tab, "Transform")

        # -------------------------
        # Deformation tab
        # -------------------------
        deform_tab = QWidget()
        deform_layout = QVBoxLayout()

        nonlinear_group = QGroupBox("TrakEM2-style sparse transform")
        nonlinear_layout = QVBoxLayout()
        self.start_nonlinear_button = QPushButton("Start Nonlinear (Shift+T)")
        self.start_nonlinear_button.clicked.connect(self.start_nonlinear_transform)
        nonlinear_layout.addWidget(self.start_nonlinear_button)

        row = QHBoxLayout()
        self.apply_nonlinear_button = QPushButton("Apply (Enter)")
        self.apply_nonlinear_button.clicked.connect(self.apply_nonlinear_current)
        row.addWidget(self.apply_nonlinear_button)
        self.cancel_nonlinear_button = QPushButton("Cancel (Esc)")
        self.cancel_nonlinear_button.clicked.connect(self.cancel_nonlinear_transform)
        row.addWidget(self.cancel_nonlinear_button)
        nonlinear_layout.addLayout(row)

        row = QHBoxLayout()
        self.propagate_first_button = QPushButton("Apply to First")
        self.propagate_first_button.clicked.connect(lambda: self.apply_nonlinear_transform("first"))
        row.addWidget(self.propagate_first_button)
        self.propagate_last_button = QPushButton("Apply to Last")
        self.propagate_last_button.clicked.connect(lambda: self.apply_nonlinear_transform("last"))
        row.addWidget(self.propagate_last_button)
        nonlinear_layout.addLayout(row)
        nonlinear_layout.addWidget(QLabel(
            "Use + then click (or Shift+click); drag red points. "
            "1=move, 2=similarity, 3=affine, 4+=local."
        ))
        nonlinear_group.setLayout(nonlinear_layout)
        deform_layout.addWidget(nonlinear_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Grid"))
        self.grid_spacing_spin = QSpinBox()
        self.grid_spacing_spin.setMinimum(10)
        self.grid_spacing_spin.setMaximum(2000)
        self.grid_spacing_spin.setValue(100)
        self.grid_spacing_spin.valueChanged.connect(self.set_grid_spacing)
        row.addWidget(self.grid_spacing_spin)
        deform_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Target in-vitro Z"))
        self.target_z_spin = QSpinBox()
        self.target_z_spin.setMinimum(0)
        self.target_z_spin.setMaximum(9999)
        self.target_z_spin.setValue(0)
        row.addWidget(self.target_z_spin)
        deform_layout.addLayout(row)

        row = QHBoxLayout()
        self.create_grid_button = QPushButton("Create Grid")
        self.create_grid_button.clicked.connect(self.create_deformation_grid)
        row.addWidget(self.create_grid_button)

        self.update_grid_button = QPushButton("Update Lines")
        self.update_grid_button.clicked.connect(self.update_deformation_grid_lines)
        row.addWidget(self.update_grid_button)
        deform_layout.addLayout(row)

        row = QHBoxLayout()
        self.apply_deform_button = QPushButton("Apply")
        self.apply_deform_button.clicked.connect(self.apply_deformation_current_slice)
        row.addWidget(self.apply_deform_button)

        self.save_deform_button = QPushButton("Save Slice")
        self.save_deform_button.clicked.connect(self.save_current_slice_deformation)
        row.addWidget(self.save_deform_button)
        deform_layout.addLayout(row)

        row = QHBoxLayout()
        self.apply_saved_deform_button = QPushButton("Apply Saved")
        self.apply_saved_deform_button.clicked.connect(self.apply_saved_deformation_current_slice)
        row.addWidget(self.apply_saved_deform_button)

        self.load_deform_button = QPushButton("Load JSON")
        self.load_deform_button.clicked.connect(self.load_deformation_json)
        row.addWidget(self.load_deform_button)
        deform_layout.addLayout(row)

        self.save_all_deform_button = QPushButton("Save JSON")
        self.save_all_deform_button.clicked.connect(self.save_all_deformations_json)
        deform_layout.addWidget(self.save_all_deform_button)

        deform_layout.addWidget(QLabel("Edit only red Destination Points. Source Points stays fixed."))
        deform_layout.addStretch(1)
        deform_tab.setLayout(deform_layout)
        tabs.addTab(deform_tab, "Deform")

        layout.addWidget(tabs)
        self.setLayout(layout)
        self._building_ui = False

        self.viewer.bind_key("Shift-T", self.start_nonlinear_transform, overwrite=True)
        self.viewer.bind_key("Enter", self.apply_nonlinear_current, overwrite=True)
        self.viewer.bind_key("Escape", self.cancel_nonlinear_transform, overwrite=True)
        self.viewer.mouse_drag_callbacks.append(self._nonlinear_mouse_callback)

        if MAGICGUI_AVAILABLE:
            print("magicgui is available, but this single-viewer version uses Qt widgets for direct slider control.")

    def make_slider_spin(self, label, min_val, max_val, default, parent_layout):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setTickInterval(15)
        slider.setTickPosition(QSlider.TicksBelow)

        spin = QSpinBox()
        spin.setMinimum(min_val)
        spin.setMaximum(max_val)
        spin.setValue(default)
        spin.setFixedWidth(70)

        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(self.update_affine_transformation)
        spin.valueChanged.connect(self.update_affine_transformation)

        row.addWidget(slider, 1)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return slider, spin

    def make_double_spin(self, label, min_val, max_val, default, parent_layout):
        row = QHBoxLayout()
        spin = self.make_double_spin_inline(label, min_val, max_val, default, row)
        parent_layout.addLayout(row)
        return spin

    def make_double_spin_inline(self, label, min_val, max_val, default, parent_layout):
        parent_layout.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setMinimum(min_val)
        spin.setMaximum(max_val)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        spin.setValue(default)
        spin.setFixedWidth(70)
        spin.valueChanged.connect(self.update_affine_transformation)
        parent_layout.addWidget(spin)
        return spin

    def toggle_3d(self):
        self.viewer.dims.ndisplay = 3 if self.ndisplay_checkbox.isChecked() else 2

    def load_image(self, layer_name):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", filter="TIFF Files (*.tif *.tiff)")
        if not file_path:
            return
        data = read_tiff(file_path)
        add_or_replace_image(self.viewer, data, layer_name, opacity=0.65 if "vitro" in layer_name else 1.0)
        self.update_target_z_range()
        print(f"Loaded {layer_name}: {file_path}")

    def load_mask(self, layer_name):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mask File", filter="TIFF Files (*.tif *.tiff)")
        if not file_path:
            return
        data = read_tiff(file_path)
        add_or_replace_labels(self.viewer, data, layer_name, opacity=0.35)
        self.update_target_z_range()
        print(f"Loaded {layer_name}: {file_path}")

    def load_images_from_config(self):
        models = CONFIG.get("models", {}) if CONFIG else {}
        pairs = [
            (INVIVO_IMAGE, models.get("invivo_slices", "")),
            (INVITRO_IMAGE, models.get("exvivo_slices", "") or models.get("invitro_slices", "")),
        ]
        for layer_name, path in pairs:
            if path and os.path.exists(path):
                data = read_tiff(path)
                add_or_replace_image(self.viewer, data, layer_name, opacity=0.65 if layer_name == INVITRO_IMAGE else 1.0)
                print(f"Loaded {layer_name} from config: {path}")
            else:
                print(f"No valid config path for {layer_name}: {path}")
        self.update_target_z_range()

    def load_masks_from_config(self):
        models = CONFIG.get("models", {}) if CONFIG else {}
        # These key names are flexible so your config does not need to be exact.
        pairs = [
            (INVIVO_MASK, models.get("invivo_mask", "") or models.get("invivo_masks", "")),
            (INVITRO_MASK, models.get("exvivo_mask", "") or models.get("exvivo_masks", "") or models.get("invitro_mask", "") or models.get("invitro_masks", "")),
        ]
        for layer_name, path in pairs:
            if path and os.path.exists(path):
                data = read_tiff(path)
                add_or_replace_labels(self.viewer, data, layer_name, opacity=0.35)
                print(f"Loaded {layer_name} from config: {path}")
            else:
                print(f"No valid config path for {layer_name}: {path}")
        self.update_target_z_range()

    def get_transform_values(self):
        return {
            "tx": self.tx_spin.value(),
            "ty": self.ty_spin.value(),
            "tz": self.tz_spin.value(),
            "rx": self.rx_spin.value(),
            "ry": self.ry_spin.value(),
            "rz": self.rz_spin.value(),
            "sx": self.sx_spin.value(),
            "sy": self.sy_spin.value(),
            "sz": self.sz_spin.value(),
        }

    def compute_current_affine(self):
        if INVITRO_IMAGE in self.viewer.layers:
            shape = self.viewer.layers[INVITRO_IMAGE].data.shape
        elif INVITRO_MASK in self.viewer.layers:
            shape = self.viewer.layers[INVITRO_MASK].data.shape
        else:
            return np.eye(4)

        if len(shape) == 2:
            # Treat 2D as z=1 for affine consistency.
            center = np.array([0, shape[0] - 1, shape[1] - 1], dtype=float) / 2
        else:
            center = (np.array(shape[:3], dtype=float) - 1) / 2

        vals = self.get_transform_values()
        return make_affine_3d(
            vals["rx"], vals["ry"], vals["rz"],
            vals["tz"], vals["ty"], vals["tx"],
            vals["sz"], vals["sy"], vals["sx"],
            center,
        )

    def update_affine_transformation(self):
        if getattr(self, "_building_ui", False):
            return
        affine = self.compute_current_affine()
        self.current_affine = affine

        for layer_name in [INVITRO_IMAGE, INVITRO_MASK, INVITRO_MATCHES]:
            if layer_name in self.viewer.layers:
                self.viewer.layers[layer_name].affine = affine
                self.viewer.layers[layer_name].refresh()

        # If a deformation grid is visible, keep it visually aligned with
        # the translated in-vitro slice. The grid stores raw coordinates
        # internally and only shifts the overlay display.
        self.refresh_deformation_grid_offset()

    def apply_saved_transform(self):
        global TRANSFORM_CONFIG
        if not TRANSFORM_CONFIG:
            QMessageBox.warning(self, "No Transform", "No transform config was loaded.")
            return

        if "parameters" in TRANSFORM_CONFIG:
            params = TRANSFORM_CONFIG["parameters"]
            self.set_transform_controls(params)
            self.update_affine_transformation()
            print("Applied saved transform parameters.")
            return

        if "affine_matrix" in TRANSFORM_CONFIG:
            affine = np.asarray(TRANSFORM_CONFIG["affine_matrix"], dtype=float)
            self.current_affine = affine
            for layer_name in [INVITRO_IMAGE, INVITRO_MASK, INVITRO_MATCHES]:
                if layer_name in self.viewer.layers:
                    self.viewer.layers[layer_name].affine = affine
                    self.viewer.layers[layer_name].refresh()
            print("Applied saved affine matrix.")
            return

        QMessageBox.warning(self, "Invalid Transform", "Transform config has no parameters or affine_matrix.")

    def set_transform_controls(self, params):
        self.tx_spin.setValue(int(params.get("tx", 0)))
        self.ty_spin.setValue(int(params.get("ty", 0)))
        self.tz_spin.setValue(int(params.get("tz", 0)))
        self.rx_spin.setValue(int(params.get("rx", 0)))
        self.ry_spin.setValue(int(params.get("ry", 0)))
        self.rz_spin.setValue(int(params.get("rz", 0)))
        self.sx_spin.setValue(float(params.get("sx", 1.0)))
        self.sy_spin.setValue(float(params.get("sy", 1.0)))
        self.sz_spin.setValue(float(params.get("sz", 1.0)))

    def save_transform(self):
        global TRANSFORM_CONFIG
        affine = self.compute_current_affine()
        TRANSFORM_CONFIG = {
            "parameters": self.get_transform_values(),
            "affine_matrix": affine.tolist(),
            "applies_to": [INVITRO_IMAGE, INVITRO_MASK, INVITRO_MATCHES],
            "axis_order": "zyx for translation/scale center, rotations are rx/ry/rz degrees",
        }

        save_path = TRANSFORM_PATH or os.path.join(os.path.dirname(__file__), "alignment_transform.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(TRANSFORM_CONFIG, f, indent=2)
        print(f"Saved transform to {save_path}")

    def reset_transform_controls(self):
        self.set_transform_controls({
            "tx": 0, "ty": 0, "tz": 0,
            "rx": 0, "ry": 0, "rz": 0,
            "sx": 1.0, "sy": 1.0, "sz": 1.0,
        })
        self.update_affine_transformation()


    def _remove_nonlinear_layers(self):
        for name in (TRAKEM_SRC_POINTS, TRAKEM_DST_POINTS):
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

    def start_nonlinear_transform(self, _event=None):
        if self.viewer.dims.ndisplay != 2:
            QMessageBox.warning(self, "2D Mode Required", "Switch to 2D display before nonlinear editing.")
            return
        try:
            self.get_deformation_target_shape()
        except RuntimeError as exc:
            QMessageBox.warning(self, "No In-vitro Layer", str(exc))
            return
        self._remove_nonlinear_layers()
        scale = get_invitro_2d_scale(self.viewer)
        empty = np.empty((0, 2), dtype=float)
        self.viewer.add_points(empty, name=TRAKEM_SRC_POINTS, size=7, face_color="gray",
                               opacity=0.45, ndim=2, scale=scale)
        self.viewer.layers[TRAKEM_SRC_POINTS].editable = False
        dst_layer = self.viewer.add_points(empty.copy(), name=TRAKEM_DST_POINTS, size=10,
                                           face_color="red", opacity=0.95, ndim=2, scale=scale)
        dst_layer.mode = "select"
        self._nonlinear_last_dst = empty.copy()
        self._syncing_nonlinear_points = False
        dst_layer.events.data.connect(self._sync_nonlinear_point_pairs)
        self._nonlinear_mode = True
        print(
            "Nonlinear mode: use the Points + tool and click to add landmarks "
            "(Shift+click also works), then drag red points."
        )

    def _sync_nonlinear_point_pairs(self, _event=None):
        """Keep immutable source landmarks paired with edits to destination points."""
        if self._syncing_nonlinear_points:
            return
        if TRAKEM_SRC_POINTS not in self.viewer.layers or TRAKEM_DST_POINTS not in self.viewer.layers:
            return
        self._syncing_nonlinear_points = True
        try:
            src_layer = self.viewer.layers[TRAKEM_SRC_POINTS]
            dst = np.asarray(self.viewer.layers[TRAKEM_DST_POINTS].data, dtype=float).reshape((-1, 2))
            src = np.asarray(src_layer.data, dtype=float).reshape((-1, 2))
            old_dst = np.asarray(getattr(self, "_nonlinear_last_dst", empty_points()), dtype=float).reshape((-1, 2))

            if len(dst) > len(src):
                # Napari appends points in add mode. Their initial destination
                # positions are also their fixed source positions.
                src = np.vstack([src, dst[len(src):]])
                src_layer.data = src
                src_layer.refresh()
            elif len(dst) < len(src):
                # Identify retained destination rows so deleting a red point
                # removes its corresponding fixed source point as well.
                retained = []
                unused = list(range(len(old_dst)))
                for point in dst:
                    if not unused:
                        break
                    distances = [np.linalg.norm(old_dst[index] - point) for index in unused]
                    retained.append(unused.pop(int(np.argmin(distances))))
                src_layer.data = src[retained] if retained else np.empty((0, 2), dtype=float)
                src_layer.refresh()
            self._nonlinear_last_dst = dst.copy()
        finally:
            self._syncing_nonlinear_points = False

    def _nonlinear_mouse_callback(self, viewer, event):
        if not self._nonlinear_mode or event.type != "mouse_press":
            return
        modifiers = [str(value).lower() for value in getattr(event, "modifiers", ())]
        if not any("shift" in value for value in modifiers):
            return
        target = next(
            (viewer.layers[name] for name in (INVITRO_IMAGE, INVITRO_MASK) if name in viewer.layers),
            None,
        )
        if target is None:
            return
        data_position = np.asarray(target.world_to_data(event.position), dtype=float)[-2:]
        display_position = self.raw_points_to_display_points(data_position)[None, :]
        layer = viewer.layers[TRAKEM_DST_POINTS]
        old = np.asarray(layer.data, dtype=float).reshape((-1, 2))
        layer.data = np.vstack([old, display_position])
        layer.refresh()
        self._sync_nonlinear_point_pairs()
        viewer.layers.selection.active = layer
        layer.mode = "select"
        yield

    def get_nonlinear_points(self):
        if TRAKEM_SRC_POINTS not in self.viewer.layers or TRAKEM_DST_POINTS not in self.viewer.layers:
            raise RuntimeError("Start nonlinear mode and add control points first.")
        src = self.display_points_to_raw_points(self.viewer.layers[TRAKEM_SRC_POINTS].data)
        dst = self.display_points_to_raw_points(self.viewer.layers[TRAKEM_DST_POINTS].data)
        if src.shape != dst.shape:
            raise RuntimeError(
                "Source/destination points became unpaired. Cancel and restart nonlinear mode."
            )
        nonlinear_transform_for_points(src, dst, self.get_deformation_target_shape())
        return src, dst

    def _apply_nonlinear_points_to_slice(self, z, src, dst):
        if INVITRO_IMAGE in self.viewer.layers:
            layer = self.viewer.layers[INVITRO_IMAGE]
            warped = warp_slice_trakem2(get_layer_slice_2d(layer, z), src, dst, order=1)
            set_layer_slice_2d(layer, z, warped)
        for name in (INVITRO_MASK, INVITRO_MATCHES):
            if name in self.viewer.layers:
                layer = self.viewer.layers[name]
                warped = warp_slice_trakem2(get_layer_slice_2d(layer, z), src, dst, order=0)
                set_layer_slice_2d(layer, z, np.rint(warped).astype(layer.data.dtype, copy=False))

    def apply_nonlinear_current(self, _event=None):
        self.apply_nonlinear_transform("current")

    def apply_nonlinear_transform(self, direction="current"):
        try:
            src, dst = self.get_nonlinear_points()
            _, model = nonlinear_transform_for_points(src, dst, self.get_deformation_target_shape())
        except RuntimeError as exc:
            QMessageBox.warning(self, "Nonlinear Transform", str(exc))
            return
        z = self.get_target_invitro_z()
        max_z = self.target_z_spin.maximum()
        if direction == "first":
            indices = range(0, z + 1)
        elif direction == "last":
            indices = range(z, max_z + 1)
        else:
            indices = (z,)
        saved_affines = {}
        try:
            for name in (INVITRO_IMAGE, INVITRO_MASK, INVITRO_MATCHES):
                if name in self.viewer.layers:
                    layer = self.viewer.layers[name]
                    saved_affines[name] = self.get_layer_affine_matrix(layer)
                    layer.affine = np.eye(4)
            for index in indices:
                self._apply_nonlinear_points_to_slice(index, src, dst)
        except Exception as exc:
            QMessageBox.warning(self, "Nonlinear Transform Failed", str(exc))
            return
        finally:
            for name, affine in saved_affines.items():
                if name in self.viewer.layers:
                    self.viewer.layers[name].affine = affine
                    self.viewer.layers[name].refresh()
        count = len(tuple(indices)) if not isinstance(indices, tuple) else len(indices)
        self.cancel_nonlinear_transform(silent=True)
        print(f"Applied {model} nonlinear transform to {count} slice(s).")

    def cancel_nonlinear_transform(self, _event=None, silent=False):
        if not self._nonlinear_mode and not any(
            name in self.viewer.layers for name in (TRAKEM_SRC_POINTS, TRAKEM_DST_POINTS)
        ):
            return
        self._remove_nonlinear_layers()
        self._nonlinear_mode = False
        if not silent:
            print("Cancelled nonlinear transform; image data was not changed.")


    def set_grid_spacing(self):
        self.grid_spacing = int(self.grid_spacing_spin.value())

    def update_target_z_range(self):
        if not hasattr(self, "target_z_spin"):
            return
        max_z = 0
        for layer_name in [INVITRO_IMAGE, INVITRO_MASK, INVITRO_MATCHES]:
            if layer_name in self.viewer.layers:
                data = self.viewer.layers[layer_name].data
                if getattr(data, "ndim", 0) >= 3:
                    max_z = max(max_z, int(data.shape[0]) - 1)
        old_value = self.target_z_spin.value()
        self.target_z_spin.setMaximum(max_z)
        self.target_z_spin.setValue(min(old_value, max_z))

    def get_target_invitro_z(self):
        self.update_target_z_range()
        if hasattr(self, "target_z_spin"):
            return int(self.target_z_spin.value())
        return current_z_index(self.viewer)

    def get_layer_affine_matrix(self, layer):
        try:
            return np.asarray(layer.affine.affine_matrix, dtype=float).copy()
        except AttributeError:
            return np.asarray(layer.affine, dtype=float).copy()

    def get_grid_display_offset_yx(self):
        """Return the y/x display offset, in point-layer data coordinates.

        The deformation grid is drawn as a 2D overlay without the in-vitro affine.
        To make it visually follow the translated in-vitro slice, shift the grid
        by the current transform's Y/X translation. The warp itself still uses
        raw in-vitro slice coordinates, so this offset is removed again before
        applying the deformation.
        """
        try:
            vals = self.get_transform_values()
            ty = float(vals.get("ty", 0.0))
            tx = float(vals.get("tx", 0.0))
        except Exception:
            ty = tx = 0.0

        sy, sx = get_invitro_2d_scale(self.viewer)
        sy = float(sy) if sy else 1.0
        sx = float(sx) if sx else 1.0
        return np.array([ty / sy, tx / sx], dtype=float)

    def raw_points_to_display_points(self, points_yx):
        return np.asarray(points_yx, dtype=float) + self.get_grid_display_offset_yx()

    def display_points_to_raw_points(self, points_yx):
        return np.asarray(points_yx, dtype=float) - self.get_grid_display_offset_yx()

    def refresh_deformation_grid_offset(self):
        """Move the source/destination/grid overlay to follow current X/Y translation.

        This keeps the visual grid aligned with the translated in-vitro slice
        without changing the saved raw deformation coordinates.
        """
        if not hasattr(self, "_grid_src_raw") or not hasattr(self, "_grid_dst_raw"):
            return

        src_display = self.raw_points_to_display_points(self._grid_src_raw)
        dst_display = self.raw_points_to_display_points(self._grid_dst_raw)

        if DEFORM_SRC_POINTS in self.viewer.layers:
            self.viewer.layers[DEFORM_SRC_POINTS].data = src_display
            self.viewer.layers[DEFORM_SRC_POINTS].refresh()
        if DEFORM_DST_POINTS in self.viewer.layers:
            self.viewer.layers[DEFORM_DST_POINTS].data = dst_display
            self.viewer.layers[DEFORM_DST_POINTS].refresh()
        self.update_deformation_grid_lines()

    def apply_deformation_with_temporary_untransform(self, z, src, dst):
        """Deform raw in-vitro slice z while preserving the visible affine transform.

        This automates the manual workflow:
        save affine -> set identity affine -> edit raw data slice -> restore affine.
        """
        layer_names = [INVITRO_IMAGE, INVITRO_MASK, INVITRO_MATCHES]
        saved_affines = {}
        try:
            for layer_name in layer_names:
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    saved_affines[layer_name] = self.get_layer_affine_matrix(layer)
                    layer.affine = np.eye(4)
                    layer.refresh()

            self.apply_deformation_points_to_slice(z, src, dst)

        finally:
            for layer_name, affine in saved_affines.items():
                if layer_name in self.viewer.layers:
                    layer = self.viewer.layers[layer_name]
                    layer.affine = affine
                    layer.refresh()

    def get_deformation_target_shape(self):
        z = self.get_target_invitro_z()
        if INVITRO_IMAGE in self.viewer.layers:
            return get_layer_slice_2d(self.viewer.layers[INVITRO_IMAGE], z).shape
        if INVITRO_MASK in self.viewer.layers:
            return get_layer_slice_2d(self.viewer.layers[INVITRO_MASK], z).shape
        raise RuntimeError("Load an in-vitro image or mask before creating a deformation grid.")

    def create_deformation_grid(self):
        try:
            shape = self.get_deformation_target_shape()
        except RuntimeError as e:
            QMessageBox.warning(self, "No In-vitro Layer", str(e))
            return

        spacing = int(self.grid_spacing_spin.value())
        try:
            ny, nx = deformation_grid_shape(shape, spacing)
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Grid", str(e))
            return
        point_count = ny * nx
        if point_count > MAX_DEFORMATION_GRID_POINTS:
            safe_spacing = minimum_grid_spacing(shape)
            QMessageBox.warning(
                self,
                "Grid Too Dense",
                f"Spacing {spacing} creates {point_count:,} control points. "
                f"Use spacing {safe_spacing} or larger (maximum "
                f"{MAX_DEFORMATION_GRID_POINTS:,} points).",
            )
            return
        points_raw, ys, xs = make_regular_grid_points(shape, spacing)
        points_display = self.raw_points_to_display_points(points_raw)

        for name in [DEFORM_SRC_POINTS, DEFORM_DST_POINTS, DEFORM_GRID]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        point_scale = get_invitro_2d_scale(self.viewer)

        self._grid_src_raw = points_raw.copy()
        self._grid_dst_raw = points_raw.copy()

        self.viewer.add_points(
            points_display,
            name=DEFORM_SRC_POINTS,
            size=6,
            face_color="gray",
            opacity=0.35,
            ndim=2,
            scale=point_scale,
        )
        self.viewer.layers[DEFORM_SRC_POINTS].editable = False

        self.viewer.add_points(
            points_display.copy(),
            name=DEFORM_DST_POINTS,
            size=8,
            face_color="red",
            opacity=0.9,
            ndim=2,
            scale=point_scale,
        )

        lines = make_grid_lines_from_points(points_display, ys, xs)
        self.viewer.add_shapes(
            lines,
            shape_type="path",
            name=DEFORM_GRID,
            edge_width=1,
            opacity=0.6,
            ndim=2,
            scale=point_scale,
        )

        self._grid_ys = ys
        self._grid_xs = xs
        self._grid_spacing = spacing
        print(
            f"Created deformation grid on the current display plane. "
            f"It will deform raw in-vitro slice z={self.get_target_invitro_z()} with {len(points_raw)} control points."
        )
        print("Move points in the Deform Destination Points layer, then click Apply.")

    def update_deformation_grid_lines(self):
        if DEFORM_SRC_POINTS not in self.viewer.layers or DEFORM_DST_POINTS not in self.viewer.layers:
            QMessageBox.warning(self, "Missing Points", "Create a deformation grid first.")
            return

        dst_display = np.asarray(self.viewer.layers[DEFORM_DST_POINTS].data, dtype=float)
        # Keep an internal raw-coordinate copy of the moved destination points.
        # This removes the visual X/Y translation offset before warping.
        self._grid_dst_raw = self.display_points_to_raw_points(dst_display)
        dst = dst_display
        if not hasattr(self, "_grid_ys") or not hasattr(self, "_grid_xs"):
            # Reconstruct a best-effort grid layout from source points.
            src_display = np.asarray(self.viewer.layers[DEFORM_SRC_POINTS].data, dtype=float)
            src = self.display_points_to_raw_points(src_display)
            self._grid_ys = sorted(set(int(round(y)) for y in src[:, 0]))
            self._grid_xs = sorted(set(int(round(x)) for x in src[:, 1]))

        lines = []
        ny = len(self._grid_ys)
        nx = len(self._grid_xs)
        if len(dst) != ny * nx:
            QMessageBox.warning(self, "Grid Changed", "Do not add/delete grid points. Move existing destination points only.")
            return

        grid = dst.reshape(ny, nx, 2)
        for i in range(ny):
            lines.append(grid[i, :, :])
        for j in range(nx):
            lines.append(grid[:, j, :])

        if DEFORM_GRID in self.viewer.layers:
            self.viewer.layers[DEFORM_GRID].data = lines
            self.viewer.layers[DEFORM_GRID].refresh()
        else:
            self.viewer.add_shapes(lines, shape_type="path", name=DEFORM_GRID, edge_width=1, opacity=0.6, ndim=2, scale=get_invitro_2d_scale(self.viewer))

    def get_current_deformation_points(self):
        if DEFORM_SRC_POINTS not in self.viewer.layers or DEFORM_DST_POINTS not in self.viewer.layers:
            raise RuntimeError("Create a deformation grid first.")

        src_display = np.asarray(self.viewer.layers[DEFORM_SRC_POINTS].data, dtype=float)
        dst_display = np.asarray(self.viewer.layers[DEFORM_DST_POINTS].data, dtype=float)

        src = self.display_points_to_raw_points(src_display)
        dst = self.display_points_to_raw_points(dst_display)

        # Store raw copies so later translation changes can move the overlay
        # without changing the actual deformation.
        self._grid_src_raw = src.copy()
        self._grid_dst_raw = dst.copy()

        src, dst = validate_deformation_points(src, dst)
        if hasattr(self, "_grid_ys") and hasattr(self, "_grid_xs"):
            expected = len(self._grid_ys) * len(self._grid_xs)
            if len(src) != expected:
                raise RuntimeError(
                    "Grid points were added or deleted. Recreate the grid and move existing red points only."
                )
        return src, dst

    def apply_deformation_current_slice(self):
        try:
            src, dst = self.get_current_deformation_points()
        except RuntimeError as e:
            QMessageBox.warning(self, "No Deformation Grid", str(e))
            return

        max_disp = float(np.max(np.linalg.norm(dst - src, axis=1)))
        if max_disp < 0.5:
            QMessageBox.information(
                self,
                "No Visible Deformation",
                "The destination points are almost identical to the source points. Move the red points farther before applying.",
            )
            return

        z = self.get_target_invitro_z()
        try:
            self.apply_deformation_with_temporary_untransform(z, src, dst)
        except Exception as e:
            QMessageBox.warning(self, "Deformation Failed", str(e))
            return
        self.save_current_slice_deformation(silent=True)
        self.update_deformation_grid_lines()
        print(
            f"Applied grid deformation to raw in-vitro slice z={z}, "
            f"then restored the current affine transform. Max point displacement: {max_disp:.2f} pixels."
        )

    def apply_deformation_points_to_slice(self, z, src, dst):
        # Image uses linear interpolation.
        if INVITRO_IMAGE in self.viewer.layers:
            image_layer = self.viewer.layers[INVITRO_IMAGE]
            image_slice = get_layer_slice_2d(image_layer, z)
            warped_image = warp_slice_with_points(image_slice, src, dst, order=1)
            set_layer_slice_2d(image_layer, z, warped_image)
            image_layer.refresh()

        # Mask and matches use nearest-neighbor interpolation to preserve labels.
        for label_layer_name in [INVITRO_MASK, INVITRO_MATCHES]:
            if label_layer_name in self.viewer.layers:
                label_layer = self.viewer.layers[label_layer_name]
                label_slice = get_layer_slice_2d(label_layer, z)
                warped_label = warp_slice_with_points(label_slice, src, dst, order=0)
                set_layer_slice_2d(label_layer, z, np.rint(warped_label).astype(label_layer.data.dtype, copy=False))
                label_layer.refresh()

    def save_current_slice_deformation(self, silent=False):
        try:
            src, dst = self.get_current_deformation_points()
        except RuntimeError as e:
            if not silent:
                QMessageBox.warning(self, "No Deformation Grid", str(e))
            return

        z = self.get_target_invitro_z()
        ny = len(getattr(self, "_grid_ys", np.unique(src[:, 0])))
        nx = len(getattr(self, "_grid_xs", np.unique(src[:, 1])))
        if ny * nx != len(src):
            if not silent:
                QMessageBox.warning(self, "Invalid Grid", "Could not determine the grid row/column layout.")
            return
        self.deformations[str(z)] = {
            "src_points_yx": src.tolist(),
            "dst_points_yx": dst.tolist(),
            "grid_spacing": int(getattr(self, "_grid_spacing", self.grid_spacing_spin.value())),
            "grid_shape": [ny, nx],
        }
        if not silent:
            print(f"Saved deformation in memory for slice z={z}.")

    def save_all_deformations_json(self):
        self.save_current_slice_deformation(silent=True)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Deformations", filter="JSON Files (*.json)")
        if not save_path:
            return
        payload = {
            "deformations": self.deformations,
            "applies_to": [INVITRO_IMAGE, INVITRO_MASK, INVITRO_MATCHES],
            "point_order": "yx napari coordinates",
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved deformations to {save_path}")

    def load_deformation_json(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Deformation JSON", filter="JSON Files (*.json)")
        if not file_path:
            return
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.deformations = payload.get("deformations", payload)
        print(f"Loaded deformations from {file_path}")

    def apply_saved_deformation_current_slice(self):
        z = self.get_target_invitro_z()
        info = self.deformations.get(str(z))
        if info is None:
            QMessageBox.warning(self, "No Saved Deformation", f"No saved deformation found for slice z={z}.")
            return

        try:
            src, dst = validate_deformation_points(
                info["src_points_yx"], info["dst_points_yx"]
            )
            grid_shape = info.get("grid_shape")
            if grid_shape is not None:
                ny, nx = (int(value) for value in grid_shape)
                if ny < 2 or nx < 2 or ny * nx != len(src):
                    raise RuntimeError("The saved grid shape does not match its control points.")
            else:
                ny = len(np.unique(src[:, 0]))
                nx = len(np.unique(src[:, 1]))
                if ny * nx != len(src):
                    raise RuntimeError("Could not reconstruct the saved grid topology.")
            self.apply_deformation_with_temporary_untransform(z, src, dst)
        except Exception as e:
            QMessageBox.warning(self, "Saved Deformation Failed", str(e))
            return

        # Show the loaded deformation grid so it can be edited further.
        for name in [DEFORM_SRC_POINTS, DEFORM_DST_POINTS, DEFORM_GRID]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self._grid_src_raw = src.copy()
        self._grid_dst_raw = dst.copy()
        src_display = self.raw_points_to_display_points(src)
        dst_display = self.raw_points_to_display_points(dst)

        point_scale = get_invitro_2d_scale(self.viewer)
        self.viewer.add_points(src_display, name=DEFORM_SRC_POINTS, size=6, face_color="gray", opacity=0.35, ndim=2, scale=point_scale)
        self.viewer.layers[DEFORM_SRC_POINTS].editable = False
        self.viewer.add_points(dst_display, name=DEFORM_DST_POINTS, size=8, face_color="red", opacity=0.9, ndim=2, scale=point_scale)

        ys = sorted(np.unique(src[:, 0]).tolist())
        xs = sorted(np.unique(src[:, 1]).tolist())
        self._grid_ys = ys
        self._grid_xs = xs
        self._grid_spacing = int(info.get("grid_spacing", self.grid_spacing_spin.value()))
        self.grid_spacing_spin.setValue(self._grid_spacing)
        self.update_deformation_grid_lines()
        print(f"Applied saved deformation for slice z={z}.")

    def save_layer(self, layer_name):
        if layer_name not in self.viewer.layers:
            QMessageBox.warning(self, "Layer Missing", f"{layer_name} is not loaded.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, f"Save {layer_name}", filter="TIFF Files (*.tif *.tiff)")
        if not save_path:
            return
        imwrite(save_path, self.viewer.layers[layer_name].data)
        print(f"Saved {layer_name} to {save_path}")

    def save_selected_layer(self):
        layer = self.viewer.layers.selection.active
        if layer is None:
            QMessageBox.warning(self, "No Selection", "No active layer selected.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, f"Save {layer.name}", filter="TIFF Files (*.tif *.tiff)")
        if not save_path:
            return
        imwrite(save_path, layer.data)
        print(f"Saved {layer.name} to {save_path}")

    def save_matches(self):
        if INVIVO_MATCHES in self.viewer.layers:
            imwrite(os.path.join(MATCHES_DIR, "invivo_matches.tif"), self.viewer.layers[INVIVO_MATCHES].data)
        if INVITRO_MATCHES in self.viewer.layers:
            imwrite(os.path.join(MATCHES_DIR, "invitro_matches.tif"), self.viewer.layers[INVITRO_MATCHES].data)
        print("Saved match layers.")


# Keep refs alive
APP_REFS = []


def main():
    load_global_config()
    load_transform_config()

    viewer = napari.Viewer(title="GlomerAlign Single Viewer", ndisplay=2)
    panel = ControlPanel(viewer)
    viewer.window.add_dock_widget(panel, name="GlomerAlign Controls", area="right")

    APP_REFS.extend([viewer, panel, panel.match_loader, panel.match_handler])
    napari.run()


if __name__ == "__main__":
    main()
