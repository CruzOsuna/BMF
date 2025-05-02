import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import griddata
import os
import re
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.path import Path  # Importar Path de matplotlib

# ========================================================================
# CONFIGURACIÓN
# ========================================================================
RADIUS_STEP = 50  
TIFF_ORIGINAL_PATH = "/media/cruz/Spatial/CycIF_human_2024/2_Visualization/t-CycIF/Images/FAHNSCC_14.ome.tiff"
X_RES = 1  # micras/píxel
Y_RES = 1

def create_sampling_area():
    """Parsea coordenadas de un archivo con formato de array de NumPy."""
    POLYGON_FILE = '/media/HDD_1/BMF/TMA_Ecology/ROI_sampling/Shapes/Square_3.txt'
    with open(POLYGON_FILE, 'r') as f:
        content = f.read()
    
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    coords = np.array(list(map(float, numbers))).reshape(-1, 2)
    return Polygon(coords)

def generate_shannon_tiff(results_csv, step, output_dir, output_tiff_name):
    """Genera el TIFF con validación de datos optimizada."""
    # Cargar solo columnas necesarias y filtrar NaNs
    df = pd.read_csv(results_csv, usecols=['center_x', 'center_y', f'step_{step}'])
    sampling_area = create_sampling_area()
    
    points = df[['center_x', 'center_y']].values
    values = df[f'step_{step}'].values

    # Filtrar puntos con valores NaN
    mask = ~np.isnan(values)
    points = points[mask]
    values = values[mask]

    if values.size == 0:
        raise ValueError("No hay datos válidos para interpolación.")
    
    print(f"Valores de Shannon (min, max): {np.min(values)}, {np.max(values)}")

    # Crear grid basado en la imagen original
    with tifffile.TiffFile(TIFF_ORIGINAL_PATH) as tif:
        width, height = tif.pages[0].shape[1], tif.pages[0].shape[0]
    
    x_coords = np.linspace(0, width * X_RES, width)
    y_coords = np.linspace(0, height * Y_RES, height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Interpolación
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.nan)

    # Crear máscara ROI con operaciones vectorizadas
    poly_coords = np.array(sampling_area.exterior.coords)
    path = Path(poly_coords)
    points_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    roi_mask = path.contains_points(points_grid).reshape(grid_x.shape)
    grid_z[~roi_mask] = np.nan

    # Rellenar NaNs con el valor mínimo
    min_val = np.min(values)
    grid_z_filled = np.nan_to_num(grid_z, nan=min_val)

    # Guardar TIFF
    output_path = os.path.join(output_dir, output_tiff_name)
    tifffile.imwrite(output_path, grid_z_filled.T)
    print(f"TIFF guardado en: {output_path}")

    # Previsualización opcional
    plt.imshow(grid_z_filled, cmap='viridis', origin='lower')
    plt.colorbar()
    plt.title("Mapa de Shannon")
    plt.show()

# ========================================================================
# Ejecución
# ========================================================================
if __name__ == "__main__":
    SAMPLE_NAME = "FAHNSCC_14"
    OUTPUT_DIR = "/media/HDD_1/ROI_Sampling-benchmark/output"
    OUTPUT_TIFF_NAME = f'shannon_map_{SAMPLE_NAME}_step_{RADIUS_STEP}.tif'
    
    results_csv = os.path.join(OUTPUT_DIR, f"shannon_index_{SAMPLE_NAME}.csv")
    generate_shannon_tiff(results_csv, RADIUS_STEP, OUTPUT_DIR, OUTPUT_TIFF_NAME)