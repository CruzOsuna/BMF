import math
import tifffile
import os
import math
import time
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from stardist.models import StarDist2D

# input path must only have the images to segment
# output path should be an empty folder
INPUT_PATH = "/media/cruz/Mice/CycIF_mice_4NQO/2_Visualization/t-CycIF/images_illumination-corrected"
OUTPUT_PATH = "/media/cruz/Mice/CycIF_mice_4NQO/3_Segmentation/Mask_Illumination-corrected/" # remember to add / in the end
# x = 8 # x number of tiles
# y = 8 # y number of tiles
arr = os.listdir(INPUT_PATH)
model_name = '2D_versatile_fluo'

if __name__ == '__main__':
    start_time = time.time()  # Capture the start time

    model = StarDist2D.from_pretrained(model_name)
    print("Image list: {}".format(arr)) # verify you only have the desired images here

    while True:
        a = input("Do you want to proceed? [y/n]")
        if a == "y":
            for image in arr:
                image_name = image
                sep = '.'
                imageid = image_name.split(sep, 1)[0]
                print("Imageid = {}".format(imageid))

                IMAGE_PATH = os.path.join(INPUT_PATH, image_name)
                print("reading image {}".format(IMAGE_PATH))
                img = tifffile.imread(IMAGE_PATH, key=0)
                print("Finish reading image {}".format(IMAGE_PATH))
                print("Image has a shape of {}".format(img.shape))

                tiles = int(math.sqrt(int(img.shape[0]*img.shape[1]/(4781712.046875))))
                print("Calculate tiles = {}".format(tiles))
                labels, _ = model.predict_instances(normalize(img), n_tiles=(tiles, tiles))
                labels = labels.astype("int32")
                output_name = OUTPUT_PATH + imageid + ".ome.tif"
                tifffile.imwrite(output_name, labels)
                print("Finish image {}".format(IMAGE_PATH))
            break  # Exit loop after processing all images

        elif a == "n":
            break
        else:
            print("Enter either [y/n]")

    end_time = time.time()  # Capture the end time
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    print(f"Script finished in {elapsed_time:.2f} seconds")  # Print the elapsed time