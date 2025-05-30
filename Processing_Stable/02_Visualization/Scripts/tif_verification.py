import os
from PIL import Image, ImageFile
from PIL import Image

# Optional: allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Increase decompression bomb warning threshold
Image.MAX_IMAGE_PIXELS = None  # ⚠️ Use with caution on untrusted files

def verify_tiff_images(base_directory):
    print(f"Checking .tif/.tiff images in: {base_directory}\n")

    errors = []
    subfolder_summary = {}
    total_subfolders = 0

    for root, dirs, files in os.walk(base_directory):
        tiff_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]

        if not tiff_files:
            continue

        subfolder_name = os.path.basename(root)
        total_subfolders += 1
        subfolder_summary[subfolder_name] = len(tiff_files)

        print(f"📁 Subfolder: {subfolder_name} ({len(tiff_files)} files)")
        for file in sorted(tiff_files):
            filepath = os.path.join(root, file)
            try:
                filesize = os.path.getsize(filepath)
                if filesize == 0:
                    errors.append(f"⚠️ Empty file: {filepath}")
                    print(f"  ⚠️ Empty → {file}")
                    continue

                with Image.open(filepath) as img:
                    img.verify()
                    print(f"  ✅ OK    → {file} ({filesize} bytes)")

            except Exception as e:
                errors.append(f"❌ Error reading {filepath}: {str(e)}")
                print(f"  ❌ Error → {file} ({str(e)})")

        print()

    # Report
    if errors:
        print("\n=== ERRORS DETECTED ===")
        for err in errors:
            print(err)
    else:
        print("\n✅ All .tif/.tiff files are in good condition.")

    print("\n=== OVERALL SUMMARY ===")
    print(f"📂 Total subfolders with .tif/.tiff files: {total_subfolders}")
    for subfolder, count in sorted(subfolder_summary.items()):
        print(f"  - {subfolder}: {count} .tif/.tiff files")

if __name__ == "__main__":
    tiff_directory = "/media/cruz/Spatial/t-CycIF_human_2025_2/02_Visualization/t-CycIF/Images_IC"
    verify_tiff_images(tiff_directory)
