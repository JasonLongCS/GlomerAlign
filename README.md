# GlomerAlign Documentation

## Overview

GlomerAlign is a Python application for aligning and comparing in vivo and ex vivo brain images, with a focus on glomeruli identification and matching. The tool provides a user-friendly interface built with Napari for visualizing, manipulating, and segmenting 3D image stacks, as well as matching corresponding structures between different imaging modalities.

## Installation

### Prerequisites

- Python 3.7+
- The following Python packages:
  - numpy
  - pandas
  - PyYAML
  - tifffile
  - scipy
  - scikit-image
  - napari
  - PyQt5

### Setup

1. Clone the repository:
   ```bash
   git clone git@github.com:CristiSoitu/GlomerAlign.git
   cd glomeralign
   ```

2. Install the required packages:
   ```bash
   pip install numpy pandas pyyaml tifffile scipy scikit-image napari pyqt5 
   ```

3. Create a configuration file:
   ```
   mkdir config
   ```
   Create a file named `config.yaml` in the `config` directory with the following structure:
   ```yaml
   models:
     invivo_slices: "path/to/invivo/slices.tif"
     invivo_segmentation: "path/to/invivo/segmentation.tif"
     exvivo_slices: "path/to/exvivo/slices.tif"
     exvivo_segmentation: "path/to/exvivo/segmentation.tif"
   ```

## Running the Application

To start GlomerAlign, simply run:

```bash
python gui.py
```

This will open two Napari viewers: one for in vivo brain images and one for ex vivo slices.

## Features

### Image Loading and Visualization

- Load TIFF images for in vivo and ex vivo data
- Load and display segmentation masks
- Save transformed or processed images

### Image Manipulation

- Select specific slices for processing
- Apply transformations to selected slices:
  - Rotation
  - Translation

### Glomeruli Matching

- Select and match corresponding structures between in vivo and ex vivo images
- Visual feedback for selected labels
- Record matches in a CSV file for later analysis
- Undo previous matches
- Save and load match data

## User Interface Guide

### Image Loader Panel

Both the in vivo and ex vivo viewers have an Image Loader panel with the following buttons:

- **Load Image**: Open a TIFF file and display it in the viewer
- **Load Mask**: Open a segmentation mask TIFF file
- **Load Matched Data**: Load previously saved matching data
- **Save Image**: Save the currently displayed image to a TIFF file
- **Select Slices**: Open a dialog to select specific slices for processing
- **Translation**: Move sliders to translate image across their respective axes
- **Rotation**: Move sliders to rotate image around their respective axes

### Keyboard Shortcuts

- **H**: Select a label in either viewer for matching
- **Z**: Undo the last match

## Matching Workflow

1. Load images in both viewers using the **Load Image** button
2. Load or create segmentation masks using the **Load Mask** button or segmentation functions
3. Click **Load Matched Data** to initialize or load existing match data
4. Select a label in the in vivo viewer by pressing **H** while hovering over it
5. Select a corresponding label in the ex vivo viewer by pressing **H**
6. The match will be recorded automatically and displayed in both viewers
7. To undo a match, press **Z** in either viewer

## Technical Implementation Details

### Main Components

1. **ImageLoader**: Handles loading, saving, and manipulating images in each viewer
2. **SliceSelectorDialog**: Provides a UI for selecting specific slices
3. **MatchHandler**: Manages the matching of structures between viewers
4. **MatchLoader**: Handles loading and saving match data

### Data Structure

The application creates a `matches` directory with the following files:

- **glomeruli.csv**: Main file recording matches between in vivo and ex vivo structures
- **invivo_matches.tif**: Image showing matched structures in the in vivo data
- **exvivo_matches.tif**: Image showing matched structures in the ex vivo data
- **invivo_glomeruli.csv**: Properties of detected structures in the in vivo data
- **exvivo_glomeruli.csv**: Properties of detected structures in the ex vivo data

### Configuration

The application uses a YAML configuration file to store paths to:

- Default in vivo and ex vivo images
- Default in vivo and ex vivo segmentation masks

## Troubleshooting

### Common Issues

1. **Missing configuration file**: Ensure the `config.yaml` file is located in the `./config/` directory
3. **Match data loading errors**: Check that both viewers have mask layers loaded

### Error Messages

- "No image loaded to select slices": You need to load an image before selecting slices
- "No slices selected for transformation": Select slices before applying transformations
- "Masks Required": Load mask layers in both viewers before attempting to load match data

## Development and Extension

### Adding New Features

1. Modify the relevant class in the code (ImageLoader, MatchHandler, etc.)
2. Add UI components to the appropriate layout
3. Connect new UI components to their handlers

### Dependencies

- **Napari**: Main visualization framework
- **PyQt**: UI components and threading
- **NumPy/SciPy**: Image processing and analysis
- **Pandas**: Data management and CSV handling

## License

[MIT]

