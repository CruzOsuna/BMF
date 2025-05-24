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
from io import BytesIO
import re

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
            "Voronoi", "Save Viewport" , "Load points", "Export cells in shape" , "Circle with n cells","Close all" 
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
    """Load shapes from text files with numpy array syntax"""
    shapes_path = Path(shapes_path)
    if not shapes_path.is_dir():
        show_info("Please select a valid directory")
        return
        
    for filename in shapes_path.glob("*.txt"):
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            # Use regex to parse numpy array syntax
            match = re.search(r'array\((.*?),\s*dtype=(\w+)\)', content, re.DOTALL)
            if not match:
                raise ValueError("Invalid numpy array format")
            
            array_str, dtype_str = match.groups()
            array_data = ast.literal_eval(array_str.strip())
            shape_array = np.array(array_data, dtype=getattr(np, dtype_str))
            
            viewer.add_shapes(
                shape_array,
                shape_type='polygon',
                edge_width=1,
                edge_color='#777777',
                face_color='transparent',
                name=filename.stem
            )
            
        except Exception as e:
            show_info(f"Error loading {filename.name}:\n{str(e)}")

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

@magicgui(call_button='Load Points', layout='vertical', points_path={"mode": "r", "filter": "*.csv"})
def load_points(points_path: Path):
    """Load sampling points layer from CSV"""
    try:
        # Read CSV with points data
        points_df = pd.read_csv(points_path)
        
        # Validate required columns
        if not {'x', 'y'}.issubset(points_df.columns):
            show_info("CSV must contain 'x' and 'y' columns")
            return

        # Extract coordinates and optional properties
        points_data = points_df[['x', 'y']].values
        properties = {
            'label': points_df['label'].tolist() if 'label' in points_df.columns else None
        }

        # Create points layer with optional text labels
        points_layer = viewer.add_points(
            points_data,
            name=points_path.stem,
            size=10,
            face_color='magenta',
            edge_color='black',
            properties=properties,
            text='label' if 'label' in points_df.columns else None
        )

        # Set initial visibility settings
        points_layer.visible = True
        show_info(f"Loaded {len(points_data)} points from {points_path.name}")

    except Exception as e:
        show_info(f"Error loading points: {str(e)}")




# -------------------------------------------------------------------------------
# Widget implementations - Circle with n cells
# -------------------------------------------------------------------------------

def circle_coordinates(cx, cy, radius, num_points=100):
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)
    return np.column_stack([x, y])

@magicgui(
    call_button='Create circle',
    layout='vertical',
    center_x={'min': -1e9, 'max': 1e9, 'step': 1},
    center_y={'min': -1e9, 'max': 1e9, 'step': 1},
    num_cells={'min': 1, 'max': 1e7, 'step': 1},
)
def create_circle_for_n_cells(
    cell_info_csv: Path = None,
    center_x: float = 0.0,
    center_y: float = 0.0,
    shape_name: str = "circle_auto",
    num_cells: int = 1000
):
    """Create a circle that contains exactly n cells from the CSV data"""
    if cell_info_csv is None or not cell_info_csv.is_file():
        show_info(f"CSV file not found: {cell_info_csv}")
        return
    
    try:
        df = pd.read_csv(cell_info_csv)
    except Exception as e:
        show_info(f"Error reading CSV: {e}")
        return

    possible_x_cols = ['X_centroid','x','X']
    possible_y_cols = ['Y_centroid','y','Y']
    
    x_col, y_col = None, None
    for c in possible_x_cols:
        if c in df.columns:
            x_col = c
            break
    for c in possible_y_cols:
        if c in df.columns:
            y_col = c
            break

    if x_col is None or y_col is None:
        show_info("No X,Y coordinate columns found in CSV.")
        return

    df['dist_to_center'] = np.sqrt((df[x_col] - center_x)**2 + (df[y_col] - center_y)**2)
    df_sorted = df.sort_values(by='dist_to_center')
    
    total_cells = len(df_sorted)
    target_num = min(num_cells, total_cells)
    if target_num < 1:
        show_info("Not enough cells or invalid cell number requested.")
        return
    
    distance_target = df_sorted.iloc[target_num - 1]['dist_to_center']
    circle_pts = circle_coordinates(cx=center_x, cy=center_y, radius=distance_target)

    existing_layer_names = [layer.name for layer in viewer.layers]
    final_name = shape_name
    if final_name in existing_layer_names:
        final_name += "_new"
    
    viewer.add_shapes(
        data=[circle_pts],
        shape_type='polygon',
        edge_color='yellow',
        face_color='blue',
        opacity=0.3,
        name=final_name
    )

    show_info(
        f"Circle created around ({center_x:.2f}, {center_y:.2f}) with radius={distance_target:.2f}.\n"
        f"Total cells included: {target_num} (of {total_cells})."
    )

# Button to pick center with click
pick_center_button = PushButton(label="Pick center with click")

def on_pick_center_click():
    """Triggered when 'Pick center with click' button is pressed"""
    show_info("Click on the image to select center...")

    def get_click(layer, event):
        """Callback that captures the first click and sets (center_x, center_y)"""
        if event.type == 'mouse_press' and event.button == 1:
            coords_world = event.position
            coords_data = layer.world_to_data(coords_world)
            
            x_clicked, y_clicked = coords_data
            create_circle_for_n_cells.center_x.value = x_clicked
            create_circle_for_n_cells.center_y.value = y_clicked

            show_info(f"Coordinates set: X={x_clicked:.2f}, Y={y_clicked:.2f}")
            layer.mouse_drag_callbacks.remove(get_click)

    if len(viewer.layers) > 0:
        image_layer = viewer.layers[0]
        image_layer.mouse_drag_callbacks.append(get_click)
    else:
        show_info("No layers available to detect click.")

pick_center_button.changed.connect(on_pick_center_click)

# Create a container widget for both the main widget and the button
def create_circle_widget():
    container = QWidget()
    layout = QVBoxLayout()
    container.setLayout(layout)
    
    layout.addWidget(create_circle_for_n_cells.native)
    layout.addWidget(pick_center_button)
    
    return container



@magicgui(
    call_button='Exportar células en Shape',
    layout='vertical',
    shape_layer={
        'label': 'Capa Shape',
        'choices': lambda _: [layer.name for layer in viewer.layers if isinstance(layer, napari.layers.Shapes)]
    },
    sample_name={
        'label': 'Nombre de muestra',
        'choices': lambda w: get_unique_samples(w.cell_info_csv.value)
    },
    output_dir={'label': 'Directorio de salida', 'mode': 'd'}
)
def export_cells_in_shape(
    cell_info_csv: Path = Path(''),
    shape_layer: str = '',
    sample_name: str = '',
    output_dir: Path = Path('.')
):
    """Exporta células dentro de una shape especificada"""
    # Validar inputs
    if not cell_info_csv.exists():
        show_info("Error: Archivo CSV no encontrado")
        return
    
    if not shape_layer or shape_layer not in viewer.layers:
        show_info("Error: Capa shape no válida")
        return
    
    try:
        # Leer CSV
        df = pd.read_csv(cell_info_csv)
        
        # Validar columnas requeridas
        required_columns = ['X_centroid', 'Y_centroid', 'Sample', 'CellID']
        if not all(col in df.columns for col in required_columns):
            show_info(f"Error: CSV debe contener columnas: {', '.join(required_columns)}")
            return
            
        # Filtrar por muestra
        sample_df = df[df['Sample'] == sample_name]
        
        if sample_df.empty:
            show_info(f"No se encontraron células para la muestra: {sample_name}")
            return
            
        # Obtener shape layer
        shape = viewer.layers[shape_layer]
        
        # Convertir coordenadas a array numpy
        points = sample_df[['X_centroid', 'Y_centroid']].values
        
        # Verificar qué puntos están dentro del shape
        inside = shape.contains(points, world=False)
        
        # Filtrar células dentro del shape
        cells_in_shape = sample_df[inside]
        
        if cells_in_shape.empty:
            show_info("No se encontraron células dentro del área seleccionada")
            return
            
        # Crear nombre de archivo
        output_path = output_dir / f"cells_in_{shape_layer}_{sample_name}.csv"
        
        # Guardar resultados
        cells_in_shape.to_csv(output_path, index=False)
        
        show_info(f"Exportadas {len(cells_in_shape)} células\nGuardado en: {output_path}")

    except Exception as e:
        show_info(f"Error: {str(e)}")

def get_unique_samples(csv_path: Path) -> list:
    """Obtiene muestras únicas del CSV"""
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
            if 'Sample' in df.columns:
                return sorted(df['Sample'].unique().tolist())
        except:
            return []
    return []





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
    "Load points": load_points,
    "Save Viewport": save_viewport,
    "Close all": close_all,
    "Circle with n cells": create_circle_widget,  # Note: Now using the container widget
    "Export cells in shape": export_cells_in_shape
}

# 2. Define tab configuration
tab_config = {
    "Input": ["Open image", "Open mask", "Load shapes", "Load points"],
    "Analysis": ["Count cells", "Metadata", "Voronoi", "Circle with n cells"],  # Added here
    "Export": ["Contrast limits", "Save shapes", "Crop ROI", "Save Viewport", "Export cells", "Export cells in shape"],
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
            # Special handling for the circle widget which is already a container
            if w == "Circle with n cells":
                viewer.window.add_dock_widget(
                    widget(),
                    name="Circle with n cells",
                    area='right',
                    allowed_areas=['right', 'left']
                )
            else:
                viewer.window.add_dock_widget(
                    widget,
                    name=widget.__name__,
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



