import napari
import zarr
import dask.array as da
import tifffile as tiff
import pandas as pd
from pathlib import Path
from magicgui import magicgui

# Initialize Napari viewer
viewer = napari.Viewer()

@magicgui(call_button='Open Image', layout='vertical')
def open_image(
    image_path: Path = Path(), 
    channel_names_csv: Path = Path(),
    contrast_limits_file: Path = Path('.')
):
    """Open a multi-channel image with channel names and optional contrast limits"""
    try:
        # Read channel names
        df_channels = pd.read_csv(channel_names_csv)
        channel_names = list(df_channels["ABS"])
    except Exception as e:
        print(f"Error reading channel names: {e}")
        channel_names = []
    
    try:
        # Open image file
        tiff_file = tiff.TiffFile(image_path)
        zarr_store = zarr.open(tiff_file.aszarr(), mode='r')
        
        # Handle pyramid images
        n_levels = len(tiff_file.series[0].levels)
        if n_levels > 1:
            pyramid = [da.from_zarr(zarr_store[i]) for i in range(n_levels)]
            multiscale = True
            n_channels = pyramid[0].shape[0]
        else:
            pyramid = da.from_zarr(zarr_store)
            multiscale = False
            n_channels = pyramid.shape[0]
            
    except Exception as e:
        print(f"Error opening image: {e}")
        return
    
    # Generate default names if needed
    if len(channel_names) != n_channels:
        print(f"Generating default names for {n_channels} channels")
        channel_names = [f"Channel_{i:03d}" for i in range(n_channels)]
    
    # Load contrast limits if provided
    contrast_limits = None
    if str(contrast_limits_file) != '.':
        try:
            with open(contrast_limits_file, 'r') as f:
                contrast_limits = eval(f.read())
        except Exception as e:
            print(f"Error loading contrast limits: {e}")
    
    # Add to viewer
    viewer.add_image(
        pyramid if multiscale else [pyramid],
        channel_axis=0,
        name=channel_names,
        contrast_limits=contrast_limits,
        multiscale=multiscale,
        visible=False
    )

@magicgui(call_button='Open Mask', layout='vertical')
def open_mask(mask_path: Path = Path()):
    """Open a segmentation mask as labels layer"""
    try:
        mask = tiff.imread(mask_path)
        
        # Handle 3D masks (Z, Y, X) by taking first slice
        if len(mask.shape) > 2 and mask.shape[0] > 1:
            mask = mask[0]
            
        viewer.add_labels(
            mask,
            name='MASK',
            opacity=0.6,
            blending='translucent'
        )
        print(f"Loaded mask with shape {mask.shape}")
        
    except Exception as e:
        print(f"Error loading mask: {e}")

# Add widgets to viewer
viewer.window.add_dock_widget(open_image)
viewer.window.add_dock_widget(open_mask)

# Start Napari
if __name__ == "__main__":
    napari.run()