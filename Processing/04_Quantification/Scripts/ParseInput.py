# Functions for parsing command line arguments for ome ilastik prep
import argparse

# Define version directly or import from package
__version__ = "1.0.0"  # Replace with actual version

def ParseInput():  # Changed from ParseInputDataExtract
    """Function for parsing command line arguments"""
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--masks', nargs='+', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--channel_names', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument(
        '--mask_props', nargs="+",
        help="""
            Space separated list of additional mask metrics.
            See skimage.measure.regionprops documentation.
        """
    )
    parser.add_argument(
        '--intensity_props', nargs="+",
        help="""
            Space separated list of intensity-based metrics.
            Includes Gini index calculation.
        """
    )
    parser.add_argument('--version', action='version', 
                      version=f'mcquant {__version__}')
    
    args = parser.parse_args()
    
    return {
        'masks': args.masks,
        'image': args.image,
        'channel_names': args.channel_names,
        'output': args.output,
        'intensity_props': set(args.intensity_props or []).union(["intensity_mean"]),
        'mask_props': args.mask_props
    }