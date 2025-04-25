''' Point annotator

ex

python point_annotator.py ../visium_data/Merfish_S2_R3.npz ../visium_data/tissue_hires_image.npz
'''

import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Force GUI backend
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from glob import glob

if __name__ == '__main__':
    print('hello world')

    parser = argparse.ArgumentParser(
        prog='point_annotator',
        description='Takes two images as input (npz format) and provides annotation interface',
    )

    parser.add_argument(
        'filename1', nargs=2,
        help='NPZ files containing "x", "y", and "I" keys'
    )
    parser.add_argument(
        'output', nargs='*',
        default=None,
        help='Output filename base'
    )

    args = parser.parse_args()

    # Load source image --------------------------------------------------------
    if args.filename1[0].isnumeric():
        ind = int(args.filename1[0])
        files = glob('/home/dtward/bmaproot/nafs/dtward/merfish/jean_fan_2021/*_rasterized.npz')
        files.sort()
        args.filename1[0] = files[ind]

    try:
        dataS = np.load(args.filename1[0])
    except Exception as e:
        raise RuntimeError(f'Failed loading source: {str(e)}')

    # Load target image --------------------------------------------------------
    if args.filename1[1].isnumeric():
        ind = int(args.filename1[1])
        files = glob('/home/dtward/bmaproot/nafs/dtward/merfish/jean_fan_2021/*_rasterized.npz')
        files.sort()
        args.filename1[1] = files[ind]

    try:
        dataT = np.load(args.filename1[1])
    except Exception as e:
        raise RuntimeError(f'Failed loading target: {str(e)}')

    # Setup output names -------------------------------------------------------
    outputS = args.filename1[0].replace('.npz', '_points.npy') if not args.output else args.output[0]
    outputT = args.filename1[1].replace('.npz', '_points.npy') if not args.output else args.output[1]

    # Prepare images -----------------------------------------------------------
    xI, yI, I = dataS['x'], dataS['y'], dataS['I'].transpose(1, 2, 0).squeeze()
    xJ, yJ, J = dataT['x'], dataT['y'], dataT['I'].transpose(1, 2, 0).squeeze()

    print(f"Source image shape: {I.shape}")
    print(f"Target image shape: {J.shape}")

    # Create figure ------------------------------------------------------------
    plt.rcParams["figure.figsize"] = (12, 8)
    fig, ax = plt.subplots(1, 2)
    
    # Plot images --------------------------------------------------------------
    ax[0].imshow(I, extent=(xI[0], xI[-1], yI[-1], yI[0]))
    ax[1].imshow(J, extent=(xJ[0], xJ[-1], yJ[-1], yJ[0]))
    ax[0].set_title('Source'), ax[1].set_title('Target')

    # Load existing annotations ------------------------------------------------
    dataS_points = {}
    try:
        dataS_points = np.load(outputS, allow_pickle=True).item()
    except: pass

    dataT_points = {}
    try:
        dataT_points = np.load(outputT, allow_pickle=True).item()
    except: pass

    # Plot existing points -----------------------------------------------------
    def plot_points(ax, points_dict):
        for name, points in points_dict.items():
            ax.scatter(*zip(*points), c='red', s=10)
            for i, (x, y) in enumerate(points):
                ax.text(
                    x, y, f'{name}{i}',
                    transform=mtransforms.offset_copy(
                        ax.transData,
                        fig=fig,
                        x=0.05,
                        y=-0.05,
                        units='inches'
                    ),
                    color='red'
                )

    plot_points(ax[0], dataS_points)
    plot_points(ax[1], dataT_points)

    # Annotation loop ----------------------------------------------------------
    plt.draw()
    plt.show(block=False)
    
    new_points = {}
    count = 0
    try:
        while True:
            name = input('Enter landmark name (blank to finish): ').strip()
            if not name: break
        
            fig.suptitle(f'Annotate {name}: Click source then target points')
            plt.pause(0.1)
            
            points = plt.ginput(n=-1, timeout=0)
            if not points: continue
            
            dataS_points[name] = points[::2]  # Even indices: source points
            dataT_points[name] = points[1::2]  # Odd indices: target points

            plot_points(ax[0], {name: dataS_points[name]})
            plot_points(ax[1], {name: dataT_points[name]})
            plt.draw()
            count += 1

    except KeyboardInterrupt:
        print("\nAnnotation interrupted")

    # Save results ------------------------------------------------------------
    np.save(outputS, dataS_points)
    np.save(outputT, dataT_points)
    print(f"Saved to:\n- {outputS}\n- {outputT}")
    plt.close()