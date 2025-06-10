import tifffile
image_path = "/home/bonem/NAS_Projects/t-CycIF/t-CycIF_human_2025/02_Visualization/t-CycIF/Images_IC/3_AGSCC-2D4_1_2.ome.tif"
with tifffile.TiffFile(image_path) as tif:
    print(f"Number of channels: {tif.series[0].shape}")
