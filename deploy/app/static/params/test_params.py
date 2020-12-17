import pandas as pd
row = pd.DataFrame(  # Metadata
    {  # DO NOT remove square brackets. They are needed for correct shape.
        'NA': [1.67],
        'PIXELSIZE': [2.4],  # um
        'RI': [11.0],
        'WAVELENGTH': [550.0],  # um
        'IMAGESIZE': [512.0],  # In case of non-square images, put lesser of the two dimensions here
        'MAGNIFICATION': [100.0],

         # Not used for real-life data. Left here for compatibility with old code. Does not need to be changed
        'ILLUMINATION_OFFCENTER_X': [0],
        'ILLUMINATION_OFFCENTER_Y': [0],
        'FRAMES': [0]
    }
)