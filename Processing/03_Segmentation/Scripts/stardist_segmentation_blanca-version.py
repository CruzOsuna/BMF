import math
import tifffile
import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from scipy.ndimage import gaussian_filter
from skimage.measure import regionprops
import matplotlib.pyplot as plt

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Parámetros ajustados para células únicas
MODEL_NAME = '2D_versatile_fluo'
INPUT_PATH = "/home/cruz/Escritorio/03_Segmentation/image"
OUTPUT_PATH = "/home/cruz/Escritorio/03_Segmentation/masks/"
SUPPORTED_EXTENSIONS = ('.tif', '.tiff')
GAUSSIAN_SIGMA = 1.5          # Ajustar según ruido en imágenes
MIN_CELL_SIZE = 512            # Tamaño mínimo en píxeles para considerar como célula
PROB_THRESHOLD = 0.7           # Umbral de probabilidad aumentado
NMS_THRESHOLD = 0.3            # Umbral de supresión no máxima reducido

def calculate_tiles(img_shape, tile_size: tuple) -> tuple:
    """Calcula división óptima de tiles basado en el modelo"""
    try:
        tiles_y = math.ceil(img_shape[0] / tile_size[0])
        tiles_x = math.ceil(img_shape[1] / tile_size[1])
        return (tiles_y, tiles_x)
    except Exception as e:
        logging.error(f"Error calculando tiles: {str(e)}")
        return (1, 1)

def process_single_cell(image_path: str, output_dir: str, model: StarDist2D) -> Optional[str]:
    """Procesa una imagen conservando solo la célula principal"""
    try:
        # Leer y pre-procesar imagen
        img = tifffile.imread(image_path, key=0)
        logging.info(f"Procesando: {image_path} (forma: {img.shape})")
        
        # Paso 1: Filtrado Gaussiano para reducir ruido
        img_filtered = gaussian_filter(img, sigma=GAUSSIAN_SIGMA)
        
        # Paso 2: Normalización más estricta
        img_normalized = normalize(img_filtered, 1, 99.8)
        
        # Paso 3: Segmentación con parámetros ajustados
        tile_size = model.config.tile_size
        n_tiles = calculate_tiles(img.shape, tile_size)
        
        labels, _ = model.predict_instances(
            img_normalized,
            n_tiles=n_tiles,
            prob_thresh=PROB_THRESHOLD,
            nms_thresh=NMS_THRESHOLD,
            min_overlap=MIN_CELL_SIZE,
            scale=(1.0, 1.0)
        )
        
        # Paso 4: Post-procesamiento para conservar solo la célula principal
        regions = regionprops(labels)
        if len(regions) > 0:
            # Seleccionar la región más grande que cumple con el tamaño mínimo
            valid_regions = [r for r in regions if r.area >= MIN_CELL_SIZE]
            if valid_regions:
                largest = max(valid_regions, key=lambda x: x.area)
                mask = np.zeros_like(labels)
                mask[largest.coords[:, 0], largest.coords[:, 1]] = 1
                labels = mask.astype("int32")
            else:
                logging.warning(f"No se encontraron células válidas en {image_path}")
                return None
        else:
            logging.warning(f"No se detectaron células en {image_path}")
            return None

        # Paso 5: Guardar y visualización
        output_path = Path(output_dir) / f"{Path(image_path).stem}_single_cell.ome.tif"
        tifffile.imwrite(output_path, labels, ome=True)
        
        # Visualización de resultados
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            fig, ax = plt.subplots(1, 2, figsize=(12,6))
            ax[0].imshow(img, cmap='gray')
            ax[0].set_title('Original')
            ax[1].imshow(labels, cmap='magma')
            ax[1].set_title('Célula Segmentada')
            plt.show()
        
        return str(output_path)

    except Exception as e:
        logging.error(f"Error procesando {image_path}: {str(e)}")
        return None

def main():
    # Configurar paths
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Directorio de entrada no encontrado: {input_path}")

    # Obtener imágenes válidas
    image_files = [
        f for f in input_path.iterdir() 
        if f.suffix.lower() in SUPPORTED_EXTENSIONS and f.is_file()
    ]

    if not image_files:
        logging.warning(f"No se encontraron imágenes válidas en {input_path}")
        return

    # Cargar modelo
    try:
        model = StarDist2D.from_pretrained(MODEL_NAME)
        logging.info(f"Modelo cargado: {MODEL_NAME}")
    except Exception as e:
        logging.error(f"Error cargando modelo: {str(e)}")
        return

    # Confirmación de usuario
    print(f"\n{'#'*40}\nIMPORTANTE: Se procesarán {len(image_files)} imágenes")
    print(f"Parámetros actuales:")
    print(f"- Filtro Gaussiano (sigma): {GAUSSIAN_SIGMA}")
    print(f"- Tamaño mínimo célula: {MIN_CELL_SIZE} pixeles")
    print(f"- Umbral probabilidad: {PROB_THRESHOLD}")
    print(f"- Umbral NMS: {NMS_THRESHOLD}\n{'#'*40}\n")
    
    if input("¿Continuar con la segmentación? [y/n]: ").lower() != 'y':
        logging.info("Proceso cancelado por el usuario")
        return

    # Procesamiento
    start_time = time.time()
    success_count = 0

    for image_file in image_files:
        result = process_single_cell(str(image_file), str(output_path), model)
        if result:
            success_count += 1
            logging.info(f"Éxito: {result}")

    # Resumen final
    elapsed_time = time.time() - start_time
    logging.info(
        f"\n{'#'*40}\nResultado final:"
        f"\n- Imágenes procesadas: {success_count}/{len(image_files)}"
        f"\n- Tiempo total: {elapsed_time:.2f} segundos"
        f"\n- Tiempo por imagen: {elapsed_time/len(image_files):.2f} segundos"
        f"\n{'#'*40}"
    )

if __name__ == '__main__':
    main()