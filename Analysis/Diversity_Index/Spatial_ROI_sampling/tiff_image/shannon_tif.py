import numpy as np
import pandas as pd
import tifffile
from scipy.interpolate import griddata
import os
import re
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt

# ========================================================================
# CONFIGURACIÓN
# ========================================================================
RADIUS_STEP = 50  
TIFF_ORIGINAL_PATH = "/media/cruz/Spatial/CycIF_human_2024/2_Visualization/t-CycIF/Images/FAHNSCC_14.ome.tiff"
X_RES = 1  # micras/píxel (ajustar según el microscopio)
Y_RES = 1

def create_sampling_area():
    """Parsea coordenadas de un archivo con formato de array de NumPy."""
    POLYGON_FILE = '/media/HDD_1/BMF/TMA_Ecology/ROI_sampling/Shapes/Square_3.txt'
    with open(POLYGON_FILE, 'r') as f:
        content = f.read()
    
    # Extraer todos los números del contenido, incluyendo negativos y decimales
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', content)
    numbers = list(map(float, numbers))
    
    # Convertir a pares (x, y)
    coords = list(zip(numbers[::2], numbers[1::2]))
    
    return Polygon(coords)

def generate_shannon_tiff(results_csv, step, output_dir, output_tiff_name):
    """Genera el TIFF con validación de datos."""
    # Cargar datos
    df = pd.read_csv(results_csv)
    sampling_area = create_sampling_area()
    points = df[['center_x', 'center_y']].values
    values = df[f'step_{step}'].values

    print(f"Valores de Shannon (min, max): {np.nanmin(values)}, {np.nanmax(values)}")

    # Crear grid basado en la imagen original
    with tifffile.TiffFile(TIFF_ORIGINAL_PATH) as tif:
        width, height = tif.pages[0].shape[1], tif.pages[0].shape[0]
    
    x_coords = np.linspace(0, width * X_RES, width)
    y_coords = np.linspace(0, height * Y_RES, height)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)

    # Interpolación y máscara del ROI
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=np.nan)
    roi_mask = np.array([sampling_area.contains(Point(x, y)) for x, y in zip(grid_x.ravel(), grid_y.ravel())])
    roi_mask = roi_mask.reshape(grid_x.shape)
    grid_z[~roi_mask] = np.nan  # Eliminar áreas fuera del ROI

    # Reemplazar NaN con el valor mínimo para visualización
    grid_z_filled = np.nan_to_num(grid_z, nan=np.nanmin(values))

    # Guardar TIFF
    output_path = os.path.join(output_dir, output_tiff_name)
    tifffile.imwrite(output_path, grid_z_filled.T)
    print(f"TIFF guardado en: {output_path}")

    # Previsualización con Matplotlib
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