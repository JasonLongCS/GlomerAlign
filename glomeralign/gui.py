import sys
import os

# Force UTF-8 on Windows consoles (cp1252 default can't encode many Unicode chars
# that napari uses in its error notifications, causing a secondary crash in the logger).
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import yaml
import json
from tifffile import imread, imwrite
from scipy.ndimage import rotate
from skimage.measure import regionprops_table
import napari
from PyQt5.QtWidgets import (
    QPushButton, QVBoxLayout, QWidget, QFileDialog, QInputDialog, QDialog,
    QScrollArea, QCheckBox, QDialogButtonBox, QHBoxLayout, QMessageBox, QLabel, QSlider
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt

# Create matches directory
MATCHES_DIR = "matches"
os.makedirs(MATCHES_DIR, exist_ok=True)

# Global config variable
CONFIG = {}
TRANSFORM_CONFIG = {}

def _get_scale(ndim):
    """Return a scale tuple (z, y, x) or (y, x) from config, defaulting to 1.0."""
    scale_cfg = CONFIG.get('scale', {})
    xy = float(scale_cfg.get('xy', 1.0))
    z = float(scale_cfg.get('z', 1.0))
    if ndim == 2:
        return (xy, xy)
    return (z, xy, xy)

def load_global_config(config_path=None):
    """Load configuration file into global CONFIG variable"""
    global CONFIG
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            CONFIG = yaml.safe_load(file)
        print(f"Config loaded from {config_path}")
        print(f"Config data: {CONFIG}")
        return True
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        CONFIG = {"models": {}}
        return False
    

    # WIP
def load_transform_config(config_path=None):
    
    global TRANSFORM_CONFIG
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "alignment_transform.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            TRANSFORM_CONFIG = json.load(file)
        print(f"Config loaded from {config_path}")
        print(f"Config data: {TRANSFORM_CONFIG}")
        return True
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        CONFIG = {"models": {}}
        return False

def apply_saved_transform(layer):
    global TRANSFORM_CONFIG

    if not TRANSFORM_CONFIG:
        print("No transform config loaded.")
        return

    #if "scale" in TRANSFORM_CONFIG:
        #layer.scale = tuple(TRANSFORM_CONFIG["scale"])

    if "affine_matrix" in TRANSFORM_CONFIG:
        layer.affine = np.asarray(TRANSFORM_CONFIG["affine_matrix"])

    layer.refresh()
    print("Applied saved affine transform.")


def rotation_matrix_3d(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

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



# Dialog for selecting slices
class SliceSelectorDialog(QDialog):
    def __init__(self, num_slices, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Slices")
        self.selected_slices = set()
        
        # Scrollable area for slices
        layout = QVBoxLayout()
        scroll_area = QScrollArea(self)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        self.checkboxes = []
        for i in range(num_slices):
            checkbox = QCheckBox(f"Slice {i}")
            checkbox.stateChanged.connect(self.update_selection)
            scroll_layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)
        
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

        # Select All / Deselect All buttons
        button_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(self.select_all)
        button_layout.addWidget(select_all_button)

        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(self.deselect_all)
        button_layout.addWidget(deselect_all_button)
        layout.addLayout(button_layout)

        # OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)

    def update_selection(self):
        self.selected_slices = {
            i for i, checkbox in enumerate(self.checkboxes) if checkbox.isChecked()
        }

    def select_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(True)

    def deselect_all(self):
        for checkbox in self.checkboxes:
            checkbox.setChecked(False)


class ImageLoader(QWidget):
    def __init__(self, viewer, viewer_type):
        super().__init__()
        self.viewer = viewer
        self.viewer_type = viewer_type  # Either 'invivo' or 'exvivo'
        self.loaded_layer_name = None  # Track the loaded image layer name
        self.selected_slices = set()

        
        # Layout
        layout = QVBoxLayout()
        
        # Buttons for loading images and masks
        self.load_image_button = QPushButton("Load Image")
        self.load_image_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_image_button)
        
        self.load_mask_button = QPushButton("Load Mask")
        self.load_mask_button.clicked.connect(self.load_mask)
        layout.addWidget(self.load_mask_button)

        self.load_matches_button = QPushButton("Load Matched Data")
        layout.addWidget(self.load_matches_button)

        # Save button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.save_image)
        layout.addWidget(self.save_button)

        # Select slices button
        self.select_slices_button = QPushButton("Select Slices")
        self.select_slices_button.clicked.connect(self.select_slices)
        layout.addWidget(self.select_slices_button)

        # Affine rotation sliders
        layout.addWidget(QLabel("Affine Rotation"))

        # add selection of current layer to this loop
        # text here

        # creating sliders for translation and rotation
        self.tx_slider = self.make_slider(-1000, 1000, 0)
        self.ty_slider = self.make_slider(-1000, 1000, 0)
        self.tz_slider = self.make_slider(-1000, 1000, 0)

        self.rx_slider = self.make_slider(-180, 180, 0)
        self.ry_slider = self.make_slider(-180, 180, 0)
        self.rz_slider = self.make_slider(-180, 180, 0)

        layout.addWidget(QLabel("Translate X"))
        layout.addWidget(self.tx_slider)
        layout.addWidget(QLabel("Translate Y"))
        layout.addWidget(self.ty_slider)
        layout.addWidget(QLabel("Translate Z"))
        layout.addWidget(self.tz_slider)

        layout.addWidget(QLabel("Rotate X"))
        layout.addWidget(self.rx_slider)
        layout.addWidget(QLabel("Rotate Y"))
        layout.addWidget(self.ry_slider)
        layout.addWidget(QLabel("Rotate Z"))
        layout.addWidget(self.rz_slider)

        self.tx_slider.valueChanged.connect(self.update_affine_transformation)
        self.ty_slider.valueChanged.connect(self.update_affine_transformation)
        self.tz_slider.valueChanged.connect(self.update_affine_transformation)
        self.rx_slider.valueChanged.connect(self.update_affine_transformation)
        self.ry_slider.valueChanged.connect(self.update_affine_transformation)
        self.rz_slider.valueChanged.connect(self.update_affine_transformation)

        # applies previously saved affine transformation from import
        self.apply_affine_button = QPushButton("Apply Saved Affine")
        self.apply_affine_button.clicked.connect(self.apply_saved_affine)
        layout.addWidget(self.apply_affine_button)

        self.setLayout(layout)
        
        # Load config data appropriate for this viewer type
        self.load_config_data()
        
    # applies saved affine transformation through the click of a button.
    def apply_saved_affine(self):
        if self.loaded_layer_name is None:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return

        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            QMessageBox.warning(self, "Warning", f"Layer '{self.loaded_layer_name}' not found.")
            return

        apply_saved_transform(layer)

    def load_config_data(self):
        """Load data from config specific to this viewer type (invivo/exvivo)"""
        global CONFIG
        
        if not CONFIG:
            print("No configuration loaded")
            return
            
        models = CONFIG.get('models', {})
        
        # Determine which data to load based on viewer type
        if self.viewer_type == 'exvivo':
            exvivo_slices = models.get('exvivo_slices', '')
            if exvivo_slices and os.path.exists(exvivo_slices):
                image_data = imread(exvivo_slices)
                self.loaded_layer_name = 'Loaded Image'
                self.viewer.add_image(image_data, name=self.loaded_layer_name, opacity=1,
                                      scale=_get_scale(image_data.ndim))
                print(f"Loaded ex vivo stack from {exvivo_slices}")

        elif self.viewer_type == 'invivo':
            invivo_slices = models.get('invivo_slices', '')
            if invivo_slices and os.path.exists(invivo_slices):
                slices = imread(invivo_slices)
                self.loaded_layer_name = 'Loaded Image'
                self.viewer.add_image(slices, name=self.loaded_layer_name, opacity=1,
                                      scale=_get_scale(slices.ndim))
                print(f"Loaded in vivo slices from {invivo_slices}")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", filter="TIFF Files (*.tif *.tiff)")
        if file_path:
            image_data = imread(file_path)
            self.loaded_layer_name = 'Loaded Image'
            self.viewer.add_image(image_data, name=self.loaded_layer_name,
                                  scale=_get_scale(3))

    def load_mask(self):
        mask_path, _ = QFileDialog.getOpenFileName(self, "Open Mask File", filter="TIFF Files (*.tif *.tiff)")
        if mask_path:
            mask_data = imread(mask_path)
            self.viewer.add_labels(mask_data, name='Mask', opacity=0.3,
                                   scale=_get_scale(3))

    def save_image(self):
        if self.loaded_layer_name is None:
            print("No image has been loaded to save.")
            return
        
        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", filter="TIFF Files (*.tif *.tiff)")
        if save_path:
            imwrite(save_path, layer.data)
            print(f"Image saved to {save_path}")

    def select_slices(self):
        if self.loaded_layer_name is None:
            print("No image loaded to select slices.")
            return
        
        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            print(f"Layer '{self.loaded_layer_name}' not found.")
            return

        # Open the slice selector dialog
        dialog = SliceSelectorDialog(num_slices=layer.data.shape[0])
        if dialog.exec_():
            self.selected_slices = dialog.selected_slices
            print(f"Selected slices: {self.selected_slices}")

    # helper method to make a slider for rotation and translation (sliding later)
    def make_slider(self, min_val, max_val, default):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(default)
        slider.setTickInterval(15)
        slider.setTickPosition(QSlider.TicksBelow)
        return slider

    # refresh based on rotation/ will rewrite based on translation
    def update_affine_transformation(self):
        if self.loaded_layer_name is None:
            return

        try:
            layer = self.viewer.layers[self.loaded_layer_name]
        except KeyError:
            return
        
        tx = self.tx_slider.value()
        ty = self.ty_slider.value()
        tz = self.tz_slider.value()

        rx = self.rx_slider.value()
        ry = self.ry_slider.value()
        rz = self.rz_slider.value()

        center = (np.array(layer.data.shape) - 1) / 2
        #change this
        layer.affine = make_affine_3d(rx, ry, rz, tz, ty, tx, center)
        layer.refresh()

class MatchHandler:
    def __init__(self, in_vivo_viewer, ex_vivo_viewer):
        self.in_vivo_viewer = in_vivo_viewer
        self.ex_vivo_viewer = ex_vivo_viewer
        self.clicked = {'in_vivo': None, 'ex_vivo': None}
        self.glomeruli_path = os.path.join(MATCHES_DIR, 'glomeruli.csv')
        self.undo_stack = []
        self.setup()

    def setup(self):
        # Bind keys for matching and undo
        self.in_vivo_viewer.bind_key('h', self.on_key_press)
        self.ex_vivo_viewer.bind_key('h', self.on_key_press)
        self.in_vivo_viewer.bind_key('z', self.undo_match)
        self.ex_vivo_viewer.bind_key('z', self.undo_match)

    def on_key_press(self, viewer):
        # Determine which viewer was used
        viewer_name = 'in_vivo' if viewer == self.in_vivo_viewer else 'ex_vivo'
        
        # Get active layer and selected label
        active_layer = viewer.layers.selection.active
        if active_layer is None or active_layer.name != 'Mask':
            print("Please select a label from the Mask layer")
            return
        #print(active_layer.data.shape)
        
        # Get the label at the current cursor position
        pos = viewer.cursor.position
        # divide the cursor position by scale to get actual matching mask labelss
        z_scale, x_scale, y_scale = _get_scale(3)
        cursor_pos = (
            pos[0] / z_scale,
            pos[1] / x_scale,
            pos[2] / y_scale,
        )
        actual_cursor_pos = tuple(map(int, np.round(cursor_pos)))
        print(actual_cursor_pos)
        try:
            selected_label = active_layer.data[actual_cursor_pos]
            if selected_label == 0:  # Background
                print("Background selected (label 0), please select a valid label")
                return
                
            self.on_label_selected(viewer_name, selected_label)
        except IndexError:
            print(f"Cursor position outside image bounds in {viewer_name}")
            
    def on_label_selected(self, viewer_name, label):
        """Store the selected label and its viewer"""
        self.clicked[viewer_name] = label
        print(f"Selected label {label} from {viewer_name} viewer")
        
        # Visual feedback
        viewer = self.in_vivo_viewer if viewer_name == 'in_vivo' else self.ex_vivo_viewer
        mask_layer = viewer.layers['Mask']
        
        # Highlight the label temporarily
        temp_data = np.zeros_like(mask_layer.data)
        temp_data[mask_layer.data == label] = 1
        viewer.add_labels(temp_data, name="Selected", opacity=0.7, scale=_get_scale(temp_data.ndim))
        #QMessageBox.information(viewer.window._qt_window, "Label Selected", 
        #                       f"Selected label {label}. Press 'm' in the other viewer to complete #match.")
        
        # Remove the highlight after a moment
        viewer.layers.remove('Selected')
        
        # Check if we can make a match
        other = 'ex_vivo' if viewer_name == 'in_vivo' else 'in_vivo'
        if self.clicked[other] is not None:
            self.record_match()

    def record_match(self):
        """Record a match between selected labels"""
        v1, v2 = self.clicked['in_vivo'], self.clicked['ex_vivo']
        if v1 is None or v2 is None:
            print("Need two labels selected.")
            return

        # Get the appropriate layers
        if 'Mask' not in self.in_vivo_viewer.layers or 'Mask' not in self.ex_vivo_viewer.layers:
            print("Mask layers not found in both viewers")
            return
            
        if 'matches' not in self.in_vivo_viewer.layers or 'matches' not in self.ex_vivo_viewer.layers:
            print("Matches layers not found. Please load matched data first.")
            return

        invivo_seg = self.in_vivo_viewer.layers['Mask'].data
        exvivo_seg = self.ex_vivo_viewer.layers['Mask'].data
        invivo_match = self.in_vivo_viewer.layers['matches'].data
        exvivo_match = self.ex_vivo_viewer.layers['matches'].data

        # Use invivo label as color for both matches
        color = v1

        # Update the matches layers
        invivo_match[invivo_seg == v1] = color
        exvivo_match[exvivo_seg == v2] = color

        # Refresh layers
        self.in_vivo_viewer.layers['matches'].refresh()
        self.ex_vivo_viewer.layers['matches'].refresh()

        # Visual feedback for successful match
        #QMessageBox.information(None, "Match Recorded", 
        #                       f"Matched invivo label {v1} with exvivo label {v2}")

        # Update CSV file
        if os.path.exists(self.glomeruli_path):
            df = pd.read_csv(self.glomeruli_path, encoding='utf-8')
        else:
            df = pd.DataFrame(columns=['invivo', 'exvivo', 'color'])
            
        # Add new match to dataframe
        df.loc[len(df)] = [v1, v2, color]
        df.to_csv(self.glomeruli_path, index=False, encoding='utf-8')

        # Save match to undo stack
        self.undo_stack.append((v1, v2, color))
        
        # Reset clicked labels
        self.clicked = {'in_vivo': None, 'ex_vivo': None}
        
        # Save updated match images
        invivo_matches_path = os.path.join(MATCHES_DIR, 'invivo_matches.tif')
        exvivo_matches_path = os.path.join(MATCHES_DIR, 'exvivo_matches.tif')
        imwrite(invivo_matches_path, invivo_match)
        imwrite(exvivo_matches_path, exvivo_match)

    def undo_match(self, viewer):
        """Undo the last match"""
        if not self.undo_stack:
            QMessageBox.information(None, "Nothing to Undo", "No matches to undo.")
            return
            
        v1, v2, color = self.undo_stack.pop()

        # Get match layers
        if 'matches' not in self.in_vivo_viewer.layers or 'matches' not in self.ex_vivo_viewer.layers:
            print("Match layers not found")
            return
            
        invivo_match = self.in_vivo_viewer.layers['matches'].data
        exvivo_match = self.ex_vivo_viewer.layers['matches'].data

        # Remove the match by setting pixels with the color back to 0
        invivo_match[invivo_match == color] = 0
        exvivo_match[exvivo_match == color] = 0
        
        # Refresh layers
        self.in_vivo_viewer.layers['matches'].refresh()
        self.ex_vivo_viewer.layers['matches'].refresh()

        # Update CSV file
        if os.path.exists(self.glomeruli_path):
            df = pd.read_csv(self.glomeruli_path, encoding='utf-8')
            # Remove the match
            df = df[~((df['invivo'] == v1) & (df['exvivo'] == v2) & (df['color'] == color))]
            df.to_csv(self.glomeruli_path, index=False, encoding='utf-8')
            
        # Save updated match images
        invivo_matches_path = os.path.join(MATCHES_DIR, 'invivo_matches.tif')
        exvivo_matches_path = os.path.join(MATCHES_DIR, 'exvivo_matches.tif')
        imwrite(invivo_matches_path, invivo_match)
        imwrite(exvivo_matches_path, exvivo_match)

        QMessageBox.information(None, "Match Undone", 
                               f"Undid match between invivo {v1} and exvivo {v2}")


class MatchLoader:
    def __init__(self, in_vivo_viewer, ex_vivo_viewer):
        self.in_vivo_viewer = in_vivo_viewer
        self.ex_vivo_viewer = ex_vivo_viewer

    def load_matches(self):
        """Load existing matches or create initial match files"""
        os.makedirs(MATCHES_DIR, exist_ok=True)
        base_path = os.path.join(MATCHES_DIR, 'glomeruli.csv')
        
        # Check for required mask layers
        if 'Mask' not in self.in_vivo_viewer.layers or 'Mask' not in self.ex_vivo_viewer.layers:
            QMessageBox.warning(None, "Masks Required", 
                              "Please load mask layers in both viewers first")
            return

        invivo_seg = self.in_vivo_viewer.layers['Mask'].data
        exvivo_seg = self.ex_vivo_viewer.layers['Mask'].data

        # Paths for match files
        invivo_matches_path = os.path.join(MATCHES_DIR, 'invivo_matches.tif')
        exvivo_matches_path = os.path.join(MATCHES_DIR, 'exvivo_matches.tif')
        invivo_glomeruli_path = os.path.join(MATCHES_DIR, 'invivo_glomeruli.csv')
        exvivo_glomeruli_path = os.path.join(MATCHES_DIR, 'exvivo_glomeruli.csv')

        if os.path.exists(base_path):
            print("Loading existing match data...")
            try:
                # Load match data
                invivo_data = imread(invivo_matches_path)
                exvivo_data = imread(exvivo_matches_path)
                QMessageBox.information(None, "Match Data Loaded", 
                                      "Loaded existing match data successfully")
            except Exception as e:
                print(f"Error loading match data: {e}")
                QMessageBox.warning(None, "Error", f"Error loading match data: {e}")
                return
        else:
            print("Creating initial match files...")
            try:
                # Create region tables
                invivo_df = self._get_region_table(invivo_seg)
                exvivo_df = self._get_region_table(exvivo_seg)
                
                # Save to CSV
                invivo_df.to_csv(invivo_glomeruli_path, index=False, encoding='utf-8')
                exvivo_df.to_csv(exvivo_glomeruli_path, index=False, encoding='utf-8')
                
                # Create empty matches CSV
                pd.DataFrame(columns=['invivo', 'exvivo', 'color']).to_csv(base_path, index=False, encoding='utf-8')
                
                # Create empty match layers
                invivo_data = np.zeros_like(invivo_seg)
                exvivo_data = np.zeros_like(exvivo_seg)
                
                # Save match TIFFs
                imwrite(invivo_matches_path, invivo_data)
                imwrite(exvivo_matches_path, exvivo_data)
                
                QMessageBox.information(None, "Match Data Created", 
                                      "Created new match data files successfully")
            except Exception as e:
                print(f"Error creating match data: {e}")
                QMessageBox.warning(None, "Error", f"Error creating match data: {e}")
                return

        # Add or update match layers in viewers
        if 'matches' in self.in_vivo_viewer.layers:
            self.in_vivo_viewer.layers['matches'].data = invivo_data
            self.in_vivo_viewer.layers['matches'].refresh()
        else:
            self.in_vivo_viewer.add_labels(invivo_data, name='matches', opacity=1.0,
                                           scale=_get_scale(invivo_data.ndim))

        if 'matches' in self.ex_vivo_viewer.layers:
            self.ex_vivo_viewer.layers['matches'].data = exvivo_data
            self.ex_vivo_viewer.layers['matches'].refresh()
        else:
            self.ex_vivo_viewer.add_labels(exvivo_data, name='matches', opacity=1.0,
                                           scale=_get_scale(exvivo_data.ndim))

    def _get_region_table(self, seg):
        """Extract region properties from segmentation"""
        # Handle 2D vs 3D data
        if seg.ndim == 2:
            props = regionprops_table(seg, properties=('label', 'centroid'))
            df = pd.DataFrame(props)
            df.columns = ['id', 'y', 'x']
            # Add z column with zeros for 2D data
            df['z'] = 0
        else:
            props = regionprops_table(seg, properties=('label', 'centroid'))
            df = pd.DataFrame(props)
            df.columns = ['id', 'z', 'y', 'x']
            
        # Add additional columns
        df['color'] = df['id']  # Use label ID as initial color
        df['matched'] = False   # Initial match status
        df['receptor'] = None   # Receptor type (to be filled later)
        
        return df


def main():
    """Main function to start the GlomerAlign application"""
    # Load configuration
    load_global_config()
    load_transform_config()

    # Create the viewers
    in_vivo_viewer = napari.Viewer(title='In Vivo Brain Viewer')
    ex_vivo_viewer = napari.Viewer(title='Ex Vivo Slices Viewer')

    # Create image loaders with viewer type specification
    in_vivo_loader = ImageLoader(in_vivo_viewer, 'invivo')
    ex_vivo_loader = ImageLoader(ex_vivo_viewer, 'exvivo')
    
    # Add dock widgets
    in_vivo_viewer.window.add_dock_widget(in_vivo_loader, name='Image Loader', area='right')
    ex_vivo_viewer.window.add_dock_widget(ex_vivo_loader, name='Image Loader', area='right')

    # Create match loader and connect to buttons
    match_loader = MatchLoader(in_vivo_viewer, ex_vivo_viewer)
    in_vivo_loader.load_matches_button.clicked.connect(match_loader.load_matches)
    ex_vivo_loader.load_matches_button.clicked.connect(match_loader.load_matches)

    # Create match handler for interactions
    match_handler = MatchHandler(in_vivo_viewer, ex_vivo_viewer)

    # Run the application
    napari.run()


if __name__ == "__main__":
    main()