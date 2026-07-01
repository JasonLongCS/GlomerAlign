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
from skimage.transform import PiecewiseAffineTransform, warp

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
DEFORMATION_PATH = None


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
    data = layer.data
    if data.ndim == 2:
        layer.data = slice_data.astype(data.dtype, copy=False)
    else:
        z = min(max(int(z), 0), data.shape[0] - 1)
        data[z] = slice_data.astype(data.dtype, copy=False)
        layer.refresh()


def make_regular_grid_points(shape_yx, spacing):
    """Return points in napari 2D coordinates, y/x order."""
    height, width = shape_yx
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


def warp_slice_with_points(slice_2d, src_yx, dst_yx, order):
    """
    Warp slice using manually moved control points.

    src_yx = original control grid positions.
    dst_yx = edited/moved control grid positions.

    The transform maps src -> dst. skimage.warp needs inverse_map, so this
    samples the original source image into the destination image.
    """
    src_xy = points_yx_to_xy(src_yx)
    dst_xy = points_yx_to_xy(dst_yx)

    tform = PiecewiseAffineTransform()
    ok = tform.estimate(src_xy, dst_xy)
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
        self._building_ui = True

        layout = QVBoxLayout()

        # Load group
        load_group = QGroupBox("Load Data")
        load_layout = QVBoxLayout()

        self.load_invivo_image_button = QPushButton("Load In-vivo Image")
        self.load_invivo_image_button.clicked.connect(lambda: self.load_image(INVIVO_IMAGE))
        load_layout.addWidget(self.load_invivo_image_button)

        self.load_invivo_mask_button = QPushButton("Load In-vivo Mask")
        self.load_invivo_mask_button.clicked.connect(lambda: self.load_mask(INVIVO_MASK))
        load_layout.addWidget(self.load_invivo_mask_button)

        self.load_invitro_image_button = QPushButton("Load In-vitro Image")
        self.load_invitro_image_button.clicked.connect(lambda: self.load_image(INVITRO_IMAGE))
        load_layout.addWidget(self.load_invitro_image_button)

        self.load_invitro_mask_button = QPushButton("Load In-vitro Mask")
        self.load_invitro_mask_button.clicked.connect(lambda: self.load_mask(INVITRO_MASK))
        load_layout.addWidget(self.load_invitro_mask_button)

        self.load_from_config_button = QPushButton("Load Images From Config")
        self.load_from_config_button.clicked.connect(self.load_images_from_config)
        load_layout.addWidget(self.load_from_config_button)

        self.load_masks_from_config_button = QPushButton("Load Masks From Config")
        self.load_masks_from_config_button.clicked.connect(self.load_masks_from_config)
        load_layout.addWidget(self.load_masks_from_config_button)

        load_group.setLayout(load_layout)
        layout.addWidget(load_group)

        # View group
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout()
        self.ndisplay_checkbox = QCheckBox("3D display")
        self.ndisplay_checkbox.setChecked(False)
        self.ndisplay_checkbox.stateChanged.connect(self.toggle_3d)
        view_layout.addWidget(self.ndisplay_checkbox)
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)

        # Match group
        match_group = QGroupBox("Matching")
        match_layout = QVBoxLayout()
        self.load_matches_button = QPushButton("Load / Initialize Matches")
        self.load_matches_button.clicked.connect(self.match_loader.load_matches)
        match_layout.addWidget(self.load_matches_button)

        self.save_matches_button = QPushButton("Save Matches")
        self.save_matches_button.clicked.connect(self.save_matches)
        match_layout.addWidget(self.save_matches_button)

        match_layout.addWidget(QLabel("Select a mask layer and press h to pick labels. Press z to undo."))
        match_group.setLayout(match_layout)
        layout.addWidget(match_group)

        # Transform group
        transform_group = QGroupBox("Transform In-vitro Image/Mask")
        transform_layout = QVBoxLayout()

        self.tx_slider, self.tx_spin = self.make_slider_spin("Translate X", -5000, 5000, 0, transform_layout)
        self.ty_slider, self.ty_spin = self.make_slider_spin("Translate Y", -5000, 5000, 0, transform_layout)
        self.tz_slider, self.tz_spin = self.make_slider_spin("Translate Z", -5000, 5000, 0, transform_layout)

        self.rx_slider, self.rx_spin = self.make_slider_spin("Rotate X", -180, 180, 0, transform_layout)
        self.ry_slider, self.ry_spin = self.make_slider_spin("Rotate Y", -180, 180, 0, transform_layout)
        self.rz_slider, self.rz_spin = self.make_slider_spin("Rotate Z", -180, 180, 0, transform_layout)

        self.sx_spin = self.make_double_spin("Scale X", 0.01, 20.0, 1.0, transform_layout)
        self.sy_spin = self.make_double_spin("Scale Y", 0.01, 20.0, 1.0, transform_layout)
        self.sz_spin = self.make_double_spin("Scale Z", 0.01, 20.0, 1.0, transform_layout)

        self.apply_saved_button = QPushButton("Apply Saved Transform")
        self.apply_saved_button.clicked.connect(self.apply_saved_transform)
        transform_layout.addWidget(self.apply_saved_button)

        self.save_transform_button = QPushButton("Save Transform")
        self.save_transform_button.clicked.connect(self.save_transform)
        transform_layout.addWidget(self.save_transform_button)

        self.reset_transform_button = QPushButton("Reset Transform Controls")
        self.reset_transform_button.clicked.connect(self.reset_transform_controls)
        transform_layout.addWidget(self.reset_transform_button)

        transform_group.setLayout(transform_layout)
        layout.addWidget(transform_group)

        # Slice deformation group
        deform_group = QGroupBox("Slice Grid Deformation")
        deform_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("Grid spacing"))
        self.grid_spacing_spin = QSpinBox()
        self.grid_spacing_spin.setMinimum(10)
        self.grid_spacing_spin.setMaximum(2000)
        self.grid_spacing_spin.setValue(100)
        self.grid_spacing_spin.valueChanged.connect(self.set_grid_spacing)
        row.addWidget(self.grid_spacing_spin)
        deform_layout.addLayout(row)

        self.create_grid_button = QPushButton("Create Grid For Current Slice")
        self.create_grid_button.clicked.connect(self.create_deformation_grid)
        deform_layout.addWidget(self.create_grid_button)

        self.update_grid_button = QPushButton("Update Grid Lines")
        self.update_grid_button.clicked.connect(self.update_deformation_grid_lines)
        deform_layout.addWidget(self.update_grid_button)

        self.apply_deform_button = QPushButton("Apply Deformation To Current Slice")
        self.apply_deform_button.clicked.connect(self.apply_deformation_current_slice)
        deform_layout.addWidget(self.apply_deform_button)

        self.save_deform_button = QPushButton("Save Current Slice Deformation")
        self.save_deform_button.clicked.connect(self.save_current_slice_deformation)
        deform_layout.addWidget(self.save_deform_button)

        self.load_deform_button = QPushButton("Load Deformation JSON")
        self.load_deform_button.clicked.connect(self.load_deformation_json)
        deform_layout.addWidget(self.load_deform_button)

        self.apply_saved_deform_button = QPushButton("Apply Saved Deformation To Current Slice")
        self.apply_saved_deform_button.clicked.connect(self.apply_saved_deformation_current_slice)
        deform_layout.addWidget(self.apply_saved_deform_button)

        self.save_all_deform_button = QPushButton("Save All Deformations JSON")
        self.save_all_deform_button.clicked.connect(self.save_all_deformations_json)
        deform_layout.addWidget(self.save_all_deform_button)

        deform_layout.addWidget(QLabel("Edit only the Destination Points layer. Source Points stays fixed."))
        deform_group.setLayout(deform_layout)
        layout.addWidget(deform_group)

        # Save group
        save_group = QGroupBox("Save Layers")
        save_layout = QVBoxLayout()
        self.save_invivo_mask_button = QPushButton("Save In-vivo Mask")
        self.save_invivo_mask_button.clicked.connect(lambda: self.save_layer(INVIVO_MASK))
        save_layout.addWidget(self.save_invivo_mask_button)

        self.save_invitro_mask_button = QPushButton("Save In-vitro Mask")
        self.save_invitro_mask_button.clicked.connect(lambda: self.save_layer(INVITRO_MASK))
        save_layout.addWidget(self.save_invitro_mask_button)

        self.save_selected_button = QPushButton("Save Selected Layer")
        self.save_selected_button.clicked.connect(self.save_selected_layer)
        save_layout.addWidget(self.save_selected_button)
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)

        self.setLayout(layout)
        self._building_ui = False

        if MAGICGUI_AVAILABLE:
            print("magicgui is available, but this single-viewer version uses Qt widgets for direct slider control.")

    def make_slider_spin(self, label, min_val, max_val, default, parent_layout):
        parent_layout.addWidget(QLabel(label))
        row = QHBoxLayout()
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

        slider.valueChanged.connect(spin.setValue)
        spin.valueChanged.connect(slider.setValue)
        slider.valueChanged.connect(self.update_affine_transformation)
        spin.valueChanged.connect(self.update_affine_transformation)

        row.addWidget(slider)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return slider, spin

    def make_double_spin(self, label, min_val, max_val, default, parent_layout):
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setMinimum(min_val)
        spin.setMaximum(max_val)
        spin.setSingleStep(0.01)
        spin.setDecimals(3)
        spin.setValue(default)
        spin.valueChanged.connect(self.update_affine_transformation)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return spin

    def toggle_3d(self):
        self.viewer.dims.ndisplay = 3 if self.ndisplay_checkbox.isChecked() else 2

    def load_image(self, layer_name):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", filter="TIFF Files (*.tif *.tiff)")
        if not file_path:
            return
        data = read_tiff(file_path)
        add_or_replace_image(self.viewer, data, layer_name, opacity=0.65 if "vitro" in layer_name else 1.0)
        print(f"Loaded {layer_name}: {file_path}")

    def load_mask(self, layer_name):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Mask File", filter="TIFF Files (*.tif *.tiff)")
        if not file_path:
            return
        data = read_tiff(file_path)
        add_or_replace_labels(self.viewer, data, layer_name, opacity=0.35)
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


    def set_grid_spacing(self):
        self.grid_spacing = int(self.grid_spacing_spin.value())

    def get_deformation_target_shape(self):
        if INVITRO_IMAGE in self.viewer.layers:
            z = current_z_index(self.viewer)
            return get_layer_slice_2d(self.viewer.layers[INVITRO_IMAGE], z).shape
        if INVITRO_MASK in self.viewer.layers:
            z = current_z_index(self.viewer)
            return get_layer_slice_2d(self.viewer.layers[INVITRO_MASK], z).shape
        raise RuntimeError("Load an in-vitro image or mask before creating a deformation grid.")

    def create_deformation_grid(self):
        try:
            shape = self.get_deformation_target_shape()
        except RuntimeError as e:
            QMessageBox.warning(self, "No In-vitro Layer", str(e))
            return

        spacing = int(self.grid_spacing_spin.value())
        points, ys, xs = make_regular_grid_points(shape, spacing)

        for name in [DEFORM_SRC_POINTS, DEFORM_DST_POINTS, DEFORM_GRID]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)

        self.viewer.add_points(
            points,
            name=DEFORM_SRC_POINTS,
            size=6,
            face_color="gray",
            opacity=0.35,
            ndim=2,
        )
        self.viewer.layers[DEFORM_SRC_POINTS].editable = False

        self.viewer.add_points(
            points.copy(),
            name=DEFORM_DST_POINTS,
            size=8,
            face_color="red",
            opacity=0.9,
            ndim=2,
        )

        lines = make_grid_lines_from_points(points, ys, xs)
        self.viewer.add_shapes(
            lines,
            shape_type="path",
            name=DEFORM_GRID,
            edge_width=1,
            opacity=0.6,
            ndim=2,
        )

        self._grid_ys = ys
        self._grid_xs = xs
        print(f"Created deformation grid for slice z={current_z_index(self.viewer)} with {len(points)} control points.")
        print("Move points in the Deform Destination Points layer, then click Apply Deformation To Current Slice.")

    def update_deformation_grid_lines(self):
        if DEFORM_DST_POINTS not in self.viewer.layers:
            QMessageBox.warning(self, "Missing Points", "Create a deformation grid first.")
            return

        dst = np.asarray(self.viewer.layers[DEFORM_DST_POINTS].data, dtype=float)
        if not hasattr(self, "_grid_ys") or not hasattr(self, "_grid_xs"):
            # Reconstruct a best-effort grid layout from source points.
            src = np.asarray(self.viewer.layers[DEFORM_SRC_POINTS].data, dtype=float)
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
            self.viewer.add_shapes(lines, shape_type="path", name=DEFORM_GRID, edge_width=1, opacity=0.6, ndim=2)

    def get_current_deformation_points(self):
        if DEFORM_SRC_POINTS not in self.viewer.layers or DEFORM_DST_POINTS not in self.viewer.layers:
            raise RuntimeError("Create a deformation grid first.")

        src = np.asarray(self.viewer.layers[DEFORM_SRC_POINTS].data, dtype=float)
        dst = np.asarray(self.viewer.layers[DEFORM_DST_POINTS].data, dtype=float)
        if src.shape != dst.shape or src.shape[0] < 3:
            raise RuntimeError("Source and destination points must have the same shape and at least 3 points.")
        return src, dst

    def apply_deformation_current_slice(self):
        try:
            src, dst = self.get_current_deformation_points()
        except RuntimeError as e:
            QMessageBox.warning(self, "No Deformation Grid", str(e))
            return

        z = current_z_index(self.viewer)
        self.apply_deformation_points_to_slice(z, src, dst)
        self.save_current_slice_deformation(silent=True)
        self.update_deformation_grid_lines()
        print(f"Applied grid deformation to in-vitro slice z={z}.")

    def apply_deformation_points_to_slice(self, z, src, dst):
        # Image uses linear interpolation.
        if INVITRO_IMAGE in self.viewer.layers:
            image_layer = self.viewer.layers[INVITRO_IMAGE]
            image_slice = get_layer_slice_2d(image_layer, z)
            warped_image = warp_slice_with_points(image_slice, src, dst, order=1)
            set_layer_slice_2d(image_layer, z, warped_image)
            image_layer.refresh()
            print(f"Def Applied grid deformation to in-vitro slice z={z}.")

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

        z = current_z_index(self.viewer)
        self.deformations[str(z)] = {
            "src_points_yx": src.tolist(),
            "dst_points_yx": dst.tolist(),
            "grid_spacing": int(self.grid_spacing_spin.value()),
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
        z = current_z_index(self.viewer)
        info = self.deformations.get(str(z))
        if info is None:
            QMessageBox.warning(self, "No Saved Deformation", f"No saved deformation found for slice z={z}.")
            return

        src = np.asarray(info["src_points_yx"], dtype=float)
        dst = np.asarray(info["dst_points_yx"], dtype=float)
        self.apply_deformation_points_to_slice(z, src, dst)

        # Show the loaded deformation grid so it can be edited further.
        for name in [DEFORM_SRC_POINTS, DEFORM_DST_POINTS, DEFORM_GRID]:
            if name in self.viewer.layers:
                self.viewer.layers.remove(name)
        self.viewer.add_points(src, name=DEFORM_SRC_POINTS, size=6, face_color="gray", opacity=0.35, ndim=2)
        self.viewer.layers[DEFORM_SRC_POINTS].editable = False
        self.viewer.add_points(dst, name=DEFORM_DST_POINTS, size=8, face_color="red", opacity=0.9, ndim=2)

        ys = sorted(set(int(round(y)) for y in src[:, 0]))
        xs = sorted(set(int(round(x)) for x in src[:, 1]))
        self._grid_ys = ys
        self._grid_xs = xs
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
