import napari
from napari_stardist import StardistWidget, StarDistPrediction, StarDistTraining
from magicgui import magic_factory
import tifffile

# ------------------------------
# 1. Cargar imágenes y máscaras
# ------------------------------
def load_data(image_path, mask_path=None):
    """Carga una imagen y su máscara (opcional) en capas de napari."""
    viewer = napari.current_viewer()
    
    # Cargar imagen
    image = tifffile.imread(image_path)
    viewer.add_image(image, name="imagen", colormap="gray")
    
    # Cargar máscara si existe
    if mask_path:
        mask = tifffile.imread(mask_path)
        viewer.add_labels(mask, name="mascara")

# ------------------------------
# 2. Inicializar el plugin de StarDist
# ------------------------------
def setup_stardist_plugin():
    viewer = napari.current_viewer()
    
    # Crear widgets de StarDist
    stardist_widget = StardistWidget(viewer)
    training_widget = StarDistTraining(viewer)
    prediction_widget = StarDistPrediction(viewer)
    
    # Añadir widgets al viewer
    viewer.window.add_dock_widget(stardist_widget, name="StarDist Main")
    viewer.window.add_dock_widget(training_widget, name="StarDist Training")
    viewer.window.add_dock_widget(prediction_widget, name="StarDist Prediction")

# ------------------------------
# 3. Ejecutar el viewer con tus datos
# ------------------------------
if __name__ == "__main__":
    # Inicializar napari
    viewer = napari.Viewer()
    
    # Cargar tus datos (ajusta las rutas)
    load_data(
        image_path="ruta/a/tu/imagen.tiff",
        mask_path="ruta/a/tu/mascara.tiff"  # Opcional
    )
    
    # Configurar el plugin
    setup_stardist_plugin()
    
    # Iniciar la aplicación
    napari.run()