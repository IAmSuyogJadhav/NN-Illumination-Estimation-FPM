import pandas as pd


illumination_params = {
    'model':'multisize_raw',  # See the list in fpm.py
    'window':'tukey',  # ['tukey', 'gaussian', None]
    'a':0.3,  # alpha, used for tukey window
    'p':10,  # p and sigma used for
    'sig':230,  # gaussian window
    'do_psd':True,  # Whether to perform PSD on the FFT before reconstructing. Helps reduce boundary artifacts.
    'starting_angle':0, # with +x axis in CW direction
    'increase_angle':True,  # Set to True if angle is increasing in CW direction as frame no. increases
    'visualize':False,
    'calibrate':True,
    'fill_empty':True,
    'tol':'auto'  # ['auto', None]
}


reconstruction_params = {
    'scale': 2,  # Scale by which to increase image size
    'window': None,  # ['tukey', 'gaussian', None]
    'a':0.3, # alpha, used for tukey window
    'p':10,  # p and sigma used for 
    'sig':230,  # gaussian window
    'do_psd': False,  # Whether to perform PSD on the FFT before reconstructing. Helps reduce boundary artifacts.
    'n_iters': 100,  # Iterations
    'do_fil': False,  # About initialization. If set to True, the initialization is filtered by multiplying with ideal aperture. Recommmended: False
    'denoise': False,  # Whether to use denoising used in the original paper during reconstruction. Usually doesn't work well as it is kinda ad hoc
    
    # Parameters from Aidukas et. al. paper
    'adaptive_noise': 1,
    'adaptive_pupil': 1,
    'adaptive_img': 1,
    'alpha': 1,
    'delta_img': 10,
    'delta_pupil': 1e-4,
    'calibrate_freq_pos': True,
    'eps': 1e-9,

    # DEBUG (Not recommended to be changed)
    'crop_our_way': True,  # Do Not Change
    'plot': False  # Used for generating figures comparing ground truth with reconstruction. Does not make sense for real-life data.
}


row = pd.DataFrame(  # Metadata
    {  # DO NOT remove square brackets. They are needed for correct interpretation.
        'NA': [0.65],
        'PIXELSIZE': [11],  # um
        'RI': [1.8],
        'WAVELENGTH': [0.450],  # um
        'IMAGESIZE': [512],  # In case of non-square images, put lesser of the two dimensions here
        'MAGNIFICATION': [100],
        
         # Not used for real-life data. Left here for compatibility with old code. Does not need to be changed
        'ILLUMINATION_OFFCENTER_X': [0],
        'ILLUMINATION_OFFCENTER_Y': [0],
        'FRAMES': [0]
    }
)


save_params = {
    'illumination': {
        'format': 'mat',  # ['mat', 'npz']
        'savedir': './illumination'  # The file will be saved as {savedir}/{tiff_file}.{format}, where tiff_file is the name of the tiff file that was passed to fpm.py
    },
    'reconstruction': {
        'format': 'png',  # ['tiff', 'png']
        'savedir': './reconstruction'  # The file will be saved as {savedir}/{tiff_file}.{format}, where tiff_file is the name of the tiff file that was passed to fpm.py
    }
}
