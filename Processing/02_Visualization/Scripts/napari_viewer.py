## Dependencies needed
import napari
from napari.layers import Shapes
from napari.utils.notifications import show_info
import pandas as pd
import numpy as np
import random
import tifffile as tiff
import scimap as sm 
from tifffile import imread
import dask.array as da
import zarr
import os
import ast
from pathlib import Path
from magicgui import magicgui
from PyQt5.QtWidgets import (
    QMessageBox, 
    QDialog, 
    QVBoxLayout, 
    QCheckBox, 
    QDialogButtonBox, 
    QApplication
)
from PyQt5.QtCore import QSettings, Qt
import sys
from dask_image.imread import imread as daskread

# Initial configuration
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Widget Selection")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        
        self.widget_list = [
            "Open image", "Open mask", "Load shapes",
            "Contrast limits", "Save shapes", "Crop ROI",
            "Count cells", "Export cells", "Metadata",
            "Voronoi", "Close all", "Save Viewport"
        ]
        
        self.settings = QSettings("MyLab", "NapariTools")
        
        layout = QVBoxLayout()
        self.checkboxes = {}
        
        for widget in self.widget_list:
            cb = QCheckBox(widget)
            cb.setChecked(self.settings.value(widget, False, type=bool))
            self.checkboxes[widget] = cb
            layout.addWidget(cb)
            
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.save_settings)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
        self.setLayout(layout)
    
    def save_settings(self):
        for widget, cb in self.checkboxes.items():
            self.settings.setValue(widget, cb.isChecked())
        self.accept()

app = QApplication.instance() or QApplication(sys.argv)
dialog = SettingsDialog()
if not dialog.exec_():
    sys.exit()

viewer = napari.Viewer()

# -------------------------------------------------------------------------------
# Widget implementations
# -------------------------------------------------------------------------------


@magicgui(
    call_button='Open image',
    layout='vertical',
    image_path={
        "label": "Image Path",
        "filter": "*.tif *.tiff *.ome.tif",
        "mode": "r"
    },
    contrast_limit_txt={
        "label": "Contrast Limits (optional)",
        "filter": "*.txt",
        "mode": "r",
        "nullable": True
    },


)
def open_large_image(image_path: Path = Path("."), 
                    contrast_limit_txt: Path = None,
                    ab_list_path: Path = None):
    """Open a multichannel image with optional parameters"""
    try:
        if not image_path.is_file():
            show_info("Please select a valid image file")
            return

        # Handle channel names
        channel_names = []
        if ab_list_path and ab_list_path.is_file():
            try:
                ab_df = pd.read_csv(ab_list_path)
                channel_names = list(ab_df["ABS"]) if "ABS" in ab_df.columns else []
            except Exception as e:
                show_info(f"Error reading channel names: {str(e)}")
                return

        # Handle contrast limits
        contrast_limits = None
        if contrast_limit_txt and contrast_limit_txt.is_file():
            try:
                with open(contrast_limit_txt, 'r') as f:
                    contrast_limits = ast.literal_eval(f.read())
            except Exception as e:
                show_info(f"Error reading contrast limits: {str(e)}")
                return

        # Load image metadata and data
        with tiff.TiffFile(image_path) as tf:
            series = tf.series[0]
            axes = series.axes

            # Determine number of channels
            if 'C' in axes:
                c_idx = axes.index('C')
                num_channels = series.shape[c_idx]
            else:
                num_channels = 1  # Single-channel image

            is_pyramidal = len(series.levels) > 1

            # Load image data
            if is_pyramidal:
                pyramid = [da.from_zarr(zarr.open(tf.aszarr(level=i))) for i in range(len(series.levels))]
            else:
                pyramid = [da.from_zarr(zarr.open(tf.aszarr()))]

            # Add channel dimension if missing
            if 'C' not in axes:
                pyramid = [level[np.newaxis, ...] for level in pyramid]

            # Generate automatic channel names if needed
            if not channel_names:
                channel_names = [f"Channel_{i+1}" for i in range(num_channels)]

        # Add image to viewer
        viewer.add_image(
            pyramid,
            channel_axis=0,
            name=channel_names,
            contrast_limits=contrast_limits,
            visible=False,
            multiscale=is_pyramidal
        )

        show_info(f"Image loaded: {image_path.name}\nChannels: {len(channel_names)}")

    except Exception as e:
        show_info(f"Critical error: {str(e)}")


@magicgui(call_button='Open mask', layout='vertical')
def open_mask(mask_path=Path()):
    seg_m = tiff.imread(mask_path)
    if (len(seg_m.shape) > 2) and (seg_m.shape[0] > 1):
        seg_m = seg_m[0]
    viewer.add_labels(seg_m, name='MASK')

@magicgui(call_button='Load Shapes', layout='vertical', shapes_path={"mode": "d"})
def load_shapes(shapes_path: Path):
    shapes_path = Path(shapes_path)
    if not shapes_path.is_dir():
        show_info("Please select a valid directory")
        return
        
    names = []
    for filename in shapes_path.glob("*.txt"):
        name = filename.stem
        names.append(name)
        with open(filename, 'r') as f:
            shapes_str = f.read()
        shapes_str = shapes_str.replace('\n', '').replace('      ', '').replace('array(', '').replace(')', '')
        shapes = ast.literal_eval(shapes_str)
        shape_arrays = [np.array(s) for s in shapes]
        viewer.add_shapes(shape_arrays, shape_type='polygon', edge_width=0,
                        edge_color='#777777ff', face_color='white', name=name)

@magicgui(call_button='Save contrast limits', layout='vertical', output_file={"mode": "d"})
def save_contrast_limits(output_file: Path, ab_list_path=Path(), name=""):
    contrast_limit = []
    ab = pd.read_csv(ab_list_path)
    ab = list(ab["ABS"])
    for antibody in ab:
        contrast_limit.append(viewer.layers[antibody].contrast_limits)

    with open(output_file / f"{name}.txt", "w") as output:
        output.write(str(contrast_limit))

@magicgui(call_button='Save shape array', layout='vertical', output_file={"mode": "d"})
def save_shapes(output_file: Path, shape_name=""):
    shapes = viewer.layers[shape_name].data
    with open(output_file / f"{shape_name}.txt", 'w') as output:
        output.write(str(shapes))

@magicgui(call_button='Cut and Save ROIs', filepath={"mode": "d"})
def cut_mask(filepath: Path, shape_name=""):
    if 'MASK' not in viewer.layers:
        show_info('No mask layer named "MASK" found')
        return
    if shape_name not in viewer.layers:
        show_info(f'No shape layer named "{shape_name}" found')
        return

    mask_to_cut = viewer.layers['MASK'].data
    shape = mask_to_cut.shape
    selected_area = viewer.layers[shape_name].to_labels(labels_shape=shape)
    removable_cells = []
    for i in range(mask_to_cut.shape[0]):
        for j in range(mask_to_cut.shape[1]):
            cell = mask_to_cut[i, j]
            if selected_area[i, j] > 0 and cell not in removable_cells and cell > 0:
                removable_cells.append(cell)
    df = pd.DataFrame({'cellid': removable_cells})
    df = df.astype(int)
    df.to_csv(filepath / f'{shape_name}_selected_cell_ids.csv', index=False)

@magicgui(call_button='Close all', layout='vertical')
def close_all():
    viewer.layers.clear()

@magicgui(call_button='View metadata', layout='vertical')
def view_metadata(adata_path=Path(), image_name="", metadata_column=""):
    path = str(adata_path)
    adata = sm.pp.mcmicro_to_scimap(path, remove_dna=False, remove_string_from_name=None, log=False,
                                random_sample=None, CellId='CellID', split='X_centroid',
                                custom_imageid=None, min_cells=None, output_dir=None)
    adata = adata[adata.obs['imageid'] == image_name]
    available_phenotypes = list(adata.obs[metadata_column].unique())
    for i in available_phenotypes:
        coordinates = adata[adata.obs[metadata_column] == i]
        coordinates = pd.DataFrame({'y': coordinates.obs["Y_centroid"], 'x': coordinates.obs["X_centroid"]})
        points = coordinates.values
        r = lambda: random.randint(0, 255)
        point_color = '#%02X%02X%02X' % (r(), r(), r())
        viewer.add_points(points, size=10, face_color=point_color, visible=False, name=i)

@magicgui(call_button='Count selected cells', layout='vertical')
def count_selected_cells(shape_name: str = "", cell_info_csv: Path = Path()):
    if 'MASK' not in viewer.layers:
        show_info('No mask layer named "MASK" found')
        return
    if shape_name not in viewer.layers:
        show_info(f'No shape layer named "{shape_name}" found')
        return

    mask_layer = viewer.layers['MASK']
    mask_data = mask_layer.data
    shape_layer = viewer.layers[shape_name]
    shape_data = shape_layer.to_labels(labels_shape=mask_data.shape)

    overlapping_cells = mask_data[shape_data > 0]
    unique_cells = np.unique(overlapping_cells)
    unique_cells = unique_cells[unique_cells != 0]
    cell_count = len(unique_cells)

    show_info(f'Total cells within "{shape_name}": {cell_count}')

@magicgui(call_button='Save cells in selected ROI', layout='vertical', output_csv={"mode": "d"})
def save_selected_cells(output_csv: Path, shape_name: str = "", cell_info_csv: Path = Path(), output_file_name: str = ""):
    if 'MASK' not in viewer.layers:
        show_info('No mask layer named "MASK" found')
        return
    if shape_name not in viewer.layers:
        show_info(f'No shape layer named "{shape_name}" found')
        return

    mask_layer = viewer.layers['MASK']
    mask_data = mask_layer.data
    shape_layer = viewer.layers[shape_name]
    shape_data = shape_layer.to_labels(labels_shape=mask_data.shape)

    overlapping_cells = mask_data[shape_data > 0]
    unique_cells = np.unique(overlapping_cells)
    unique_cells = unique_cells[unique_cells != 0]
    cell_count = len(unique_cells)

    show_info(f'Total cells within "{shape_name}": {cell_count}')

    try:
        cell_info_df = pd.read_csv(cell_info_csv)
    except Exception as e:
        show_info(f'Error reading cell information file: {e}')
        return

    cell_id_column = None
    for col in ['cellid', 'CellID', 'cell_id', 'Cell_Id', 'cellID']:
        if col in cell_info_df.columns:
            cell_id_column = col
            break
    if cell_id_column is None:
        show_info('No cell ID column found in cell information file')
        return

    selected_cells_info = cell_info_df[cell_info_df[cell_id_column].isin(unique_cells)]

    try:
        selected_cells_info.to_csv(output_csv / f"{output_file_name}.csv", index=False)
        show_info(f'Information on {cell_count} selected cells saved in {output_csv}')
    except Exception as e:
        show_info(f'Error saving selected cells file: {e}')

@magicgui(call_button='Voronoi plot', layout='vertical', output_dir={"mode": "d"})
def voronoi_plot(output_dir: Path, adata_path=Path(), shape_name="", image_name="", cluster_name="", file_name=""):
    path = str(adata_path)
    adata = sm.pp.mcmicro_to_scimap(path, remove_dna=False, remove_string_from_name=None, log=False,
                                    random_sample=None, CellId='CellID', split='X_centroid',
                                    custom_imageid=None, min_cells=None, output_dir=None)
    shapes = viewer.layers[shape_name].data
    shapes = shapes[0].tolist()
    x = shapes[0]
    y = shapes[2]
    x_1 = x[1]
    x_2 = y[1]
    y_1 = x[0]
    y_2 = y[0]
    n_colors = {0: "#D3D3D3", 1: '#D3D3D3', 2: "#D3D3D3", 3: "#FF0000", 4: "#D3D3D3",
                5: "#D3D3D3", 6: '#D3D3D3', 7: "#FFD343", 8: "#D3D3D3", 9: "#D3D3D3"}
    sm.pl.voronoi(adata, color_by=cluster_name, x_coordinate='X_centroid', y_coordinate='Y_centroid', imageid='imageid',
                  subset=image_name, x_lim=[x_1, x_2], y_lim=[y_1, y_2], plot_legend=True, flip_y=True,
                  overlay_points=cluster_name, voronoi_alpha=0.7, voronoi_line_width=0.3, overlay_point_size=8,
                  overlay_point_alpha=1, legend_size=15, overlay_points_colors=n_colors, colors=n_colors,
                  fileName=f"{file_name}.pdf", saveDir=str(output_dir))


@magicgui(
    call_button='Save Viewport',
    layout='vertical',
    output_dir={"label": "Output Directory", "mode": "d"},
    filename={"label": "Filename", "tooltip": "Without extension"},
    image_layer={
        "label": "Image Layer", 
        "choices": lambda _: [layer.name for layer in viewer.layers if isinstance(layer, napari.layers.Image)]
    }
)
def save_viewport(
    output_dir: Path = Path.home(),
    filename: str = "viewport_snapshot",
    image_layer: str = None
):
    """Save current field of view as TIFF"""
    try:
        if not image_layer:
            show_info("Please select an image layer")
            return
            
        layer = viewer.layers[image_layer]
        
        # Get current view parameters
        view = viewer.window.qt_viewer
        canvas_size = view.canvas.size
        camera_zoom = view.camera.zoom

        # Calculate visible area in data coordinates
        transform = layer._transforms[0:2]  # Get spatial transforms
        visible_rect = view.camera.rect
        top_left = transform.inverse(visible_rect.top_left)
        bottom_right = transform.inverse(visible_rect.bottom_right)

        # Convert to pixel coordinates
        y_start = int(max(0, top_left[0]))
        y_end = int(min(layer.data.shape[-2], bottom_right[0]))
        x_start = int(max(0, top_left[1]))
        x_end = int(min(layer.data.shape[-1], bottom_right[1]))

        # Handle multiscale images
        if layer.multiscale:
            # Calculate optimal pyramid level
            base_scale = layer.data[0].shape[-2:]
            scale_factors = [
                (base_scale[0]/level_data.shape[-2], 
                base_scale[1]/level_data.shape[-1]
            ) for level_data in layer.data
            ]
            
            # Find level closest to current zoom
            target_scale = 1 / camera_zoom
            level = np.argmin([
                abs((sf[0] + sf[1])/2 - target_scale) 
                for sf in scale_factors
            ])
            
            data = layer.data[level]
            sf_y, sf_x = scale_factors[level]

            # Adjust coordinates for pyramid level
            y_start = int(y_start / sf_y)
            y_end = int(y_end / sf_y)
            x_start = int(x_start / sf_x)
            x_end = int(x_end / sf_x)
        else:
            data = layer.data

        # Extract viewport data
        if data.ndim == 2:
            viewport = data[y_start:y_end, x_start:x_end]
        elif data.ndim == 3:  # Handle CYX format
            viewport = data[:, y_start:y_end, x_start:x_end]
        else:
            show_info("Unsupported image dimensions")
            return

        # Save TIFF
        output_path = output_dir / f"{filename}.tiff"
        tiff.imwrite(output_path, viewport)
        show_info(f"Viewport saved:\n{output_path.name}")

    except Exception as e:
        show_info(f"Error saving viewport: {str(e)}")

# -------------------------------------------------------------------------------
# Final configuration
# -------------------------------------------------------------------------------

# 1. Define widget mapping
widget_map = {
    "Open image": open_large_image,
    "Open mask": open_mask,
    "Load shapes": load_shapes,
    "Contrast limits": save_contrast_limits,
    "Save shapes": save_shapes,
    "Crop ROI": cut_mask,
    "Count cells": count_selected_cells,
    "Export cells": save_selected_cells,
    "Metadata": view_metadata,
    "Voronoi": voronoi_plot,
    "Close all": close_all,
    "Save Viewport": save_viewport  # Added entry
}


# 2. Define tab configuration
tab_config = {
    "Input": ["Open image", "Open mask", "Load shapes"],
    "Analysis": ["Count cells", "Metadata", "Voronoi"],
    "Export": ["Contrast limits", "Save shapes", "Crop ROI", "Save Viewport", "Export cells"],
    "Tools": ["Close all"]
}


# 3. Add widgets to viewer
for tab_name, widgets in tab_config.items():
    tab_widgets = []
    for w in widgets:
        if w in dialog.checkboxes and dialog.checkboxes[w].isChecked():
            tab_widgets.append(widget_map[w])
    
    if tab_widgets:
        for widget in tab_widgets:
            viewer.window.add_dock_widget(
                widget,
                name=tab_name,
                area='right',
                allowed_areas=['right', 'left']
            )

@magicgui(call_button='⚙️ Configure Widgets')
def config_widgets():
    dialog = SettingsDialog()
    if dialog.exec_():
        viewer.window._dock_widgets.clear()
        for tab_name, widgets in tab_config.items():
            current_widgets = []
            for w in widgets:
                if dialog.checkboxes[w].isChecked():
                    current_widgets.append(widget_map[w])
            
            if current_widgets:
                for widget in current_widgets:
                    viewer.window.add_dock_widget(
                        widget,
                        name=tab_name,
                        area='right',
                        allowed_areas=['right', 'left']
                    )

viewer.window.add_dock_widget(config_widgets, area='right')

napari.run()



