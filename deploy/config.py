import os
import random
from flask import url_for
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = 'secret'
    TEMPLATES_AUTO_RELOAD = True
    UPLOAD_FOLDER = './app/static/images'
    PARAMS_FOLDER = './app/static/params'
    MODELS_FOLDER = './app/static/models'
    OUTPUT_FOLDER = './app/static/output'

    MODELS = {
        '64_raw': ('./app/static/models/frcnn_64_raw', False),
        '64_pre': ('./app/static/models/frcnn_64_pre', True),
        '128_raw': ('./app/static/models/frcnn_128_raw', False),
        '128_pre': ('./app/static/models/frcnn_128_pre', True),
        '256_raw': ('./app/static/models/frcnn_256_raw', False),
        '256_pre': ('./app/static/models/frcnn_256_pre', True),
        '512_raw': ('./app/static/models/frcnn_512_raw', False),
        '512_pre': ('./app/static/models/frcnn_512_pre', True),
        'multisize_raw': ('./app/static/models/frcnn_multisize_raw', False),
    }

    MODELS_LIST = MODELS.keys()
    EXPORT_FOLDER = './app/static/images/exported'
    EXPORT_FOLDER_REL = 'images/exported/'
    MAX_CONTENT_PATH = 5e6
    NO_MODEL = True

    DEBUG = True
    MODEL = None
    LOADED_MODEL = None
    WORKING_FILE = None

    PARAMS = None

    ILL_PARAMS = {
        'model': 'multisize_raw',  # See the list in fpm.py
        'window': 'tukey',  # ['tukey', 'gaussian', None]
        'a': 0.3,  # alpha, used for tukey window
        'p': 10,  # p and sigma used for
        'sig': 230,  # gaussian window
        'do_psd': True,  # Whether to perform PSD on the FFT before reconstructing. Helps reduce boundary artifacts.
        'starting_angle': 0,  # with +x axis in CW direction
        'increase_angle': True,  # Set to True if angle is increasing in CW direction as frame no. increases
        'visualize': False,
        'calibrate': True,
        'fill_empty': True,
        'tol': 'auto'  # ['auto', None]
    }

    REC_PARAMS = {
        'scale': 2,  # Scale by which to increase image size
        'window': None,  # ['tukey', 'gaussian', None]
        'a': 0.3, # alpha, used for tukey window
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

    SAVE_PARAMS = {
        'illumination': {
            'format': 'mat',  # ['mat', 'npz']
            'savedir': os.path.join(OUTPUT_FOLDER, 'illumination')  # The file will be saved as {savedir}/{tiff_file}.{format}, where tiff_file is the name of the tiff file that was passed to fpm.py
        },
        'reconstruction': {
            'format': 'tiff',  # ['tiff', 'png']
            'savedir': os.path.join(OUTPUT_FOLDER, 'reconstruction')  # The file will be saved as {savedir}/{tiff_file}.{format}, where tiff_file is the name of the tiff file that was passed to fpm.py
        }
    }
