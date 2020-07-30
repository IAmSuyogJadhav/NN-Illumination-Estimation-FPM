import os
import cv2
from sys import exit
import pickle
import numpy as np
import cupy as cp
import skimage.io
from tqdm.auto import tqdm
from scipy import signal
import scipy.io as io
from scipy.signal import general_gaussian
from psd import periodic_smooth_decomp as psd
from utils import *
from fpm_utils import *
from params import illumination_params, save_params, row, reconstruction_params

models = {
    '64_raw': ('models/frcnn_64_raw', False),
    '64_pre': ('models/frcnn_64_pre', True),
    '128_raw': ('models/frcnn_128_raw', False),
    '128_pre': ('models/frcnn_128_pre', True),
    '256_raw': ('models/frcnn_256_raw', False),
    '256_pre': ('models/frcnn_256_pre', True),
    '512_raw': ('models/frcnn_512_raw', False),
    '512_pre': ('models/frcnn_512_pre', True),
    'multisize_raw': ('models/frcnn_multisize_raw', False),
}


def get_illumination(tiff_path, model='multisize_raw', window='tukey', a=0.3, p=10, sig=230, do_psd=True, starting_angle=0, increase_angle=True, visualize=False, calibrate=True, fill_empty=True, tol='auto'):
    """
    get_illumination(model)
    
    Parameters
    ----------
    `tiff_path`: String, Required.
            Path to the tiff file.
    `model`: String, Optional.
        Which model to use. Defaults to multisize_raw. Currently available: 
        ['64_raw', '64_pre', '128_raw', '128_pre', '256_raw', '256_pre', '512_raw', '512_pre', 'multisize_raw']
    `window`: String, Optional
        Which window to use for apodization before estimating illumination. Defaults to 'tukey'.
        Available options: ['gaussian', 'tukey', None]
    `a, p, sig`: Floats, optional
        Window parameters. `a` is alpha used for generating tukey window.
        `p` and `sig` are used for gaussian window.
    `do_psd`: Boolean, Optional
        Whether to perform periodic smooth decomposition before estimating illumination.
        Helps in removing artifacts. Defaults to True.
    `starting_angle`: Float, Optional
        If the starting angle is not at 0 degrees measured clockwise from +x axis, change this to whatever the starting angle is in degrees.
        Must be a positive value between 0-360.
    `increase_angle`: Boolean, Optional
        Defaults to True. If the clockwise angle from +x axis does not go on increasing with frame no., set this to False.
    `visualize`: Boolean, Optional
        Defaults to False. Whether to visualize te disc locations using a matplotlib plot.
    `calibrate`: Boolean, Optional
        Defaults to True. Whether to use calibration or not. 
    `fill_empty`: Boolean, Optional
        Defaults to True. Whether to fill frames that have no detection with estimated disc locations. 
    `tol`: String, Optional
        Defaults to 'auto'. Tolerance value for rejecting wrong detections. If set to None or 'none', does not reject any detection.
    
    """
    
    assert model in models.keys(), "The model must be in the format {size}_{raw/pre}. For example:"\
    " 128_pre for model trained on 128x128 preprocessed frames. Available models:\n"\
    f"{models.keys()}"
    
    # Load the model
    print('Loading the model...', end='')
    cfg_path, preprocess_fft = models[model]
    cfg = pickle.load(open(os.path.join(cfg_path, 'cfg.pkl'), 'rb'))
    cfg.MODEL.WEIGHTS = os.path.join(cfg_path, 'model_final.pth')
    predictor = Predictor(cfg, preprocess_fft=preprocess_fft)
    print('Done!')
    
    # Read the tiff file
    imgs = read_tiff(tiff_path)
    
    if imgs[0].shape[0] != imgs[0].shape[1]:
        print('The input image is not square. The image will be cropped to be a*a where `a` is the smallest dimension.')
        a = min(imgs[0].shape[0], imgs[0].shape[1])
        slice_x, slice_y = slice(0, a), slice(0, a)
        imgs = [img[slice_x, slice_y] for img in imgs]

    width, height = imgs[0].shape  # Images aren't supposed to have 3rd dimension
    
    # Apodization
    if window.lower()=='gaussian':
        w = np.outer(signal.general_gaussian(512, p=p, sig=sig), signal.general_gaussian(512, p=p, sig=sig))
    elif window.lower()=='tukey':
        w = np.outer(signal.tukey(width, alpha=a), signal.tukey(height, alpha=a))
    elif window is None or window.lower() is 'none':
        w=1

    imgs = [w*img for img in tqdm(imgs, desc='Processing Apodization', leave=False)]
    
    # Periodic Smooth Decomposition
    if do_psd:
        imgs = [psd(img)[0] for img in tqdm(imgs, desc='Processing PSD', leave=False)]
    
    # Figure out idxs
    if (starting_angle == 0) and increase_angle:  # The default case
        idxs = None
    else:
        thetas = list(np.arange(0, 360, 360/39))
        offset = int((starting_angle / 360) * len(imgs))  # Calculate offset index
        if increase_angle:  # If angle increasing in CW direction
            thetas_ = thetas[offset:] + thetas[:offset]
            idxs = [thetas.index(t) for t in thetas_]
        else:
            thetas_ = thetas[offset::-1] + thetas[- (len(imgs) - offset -1):][::-1]
            idxs = [thetas.index(t) for t in thetas_]
    
    # Estimate illumination
    try:
        discs, radii = predictor.get_disc(
            tiff_path=imgs,
            visualize=visualize,
            warnings=False,
            calibrate=calibrate,
            fill_empty=fill_empty,
            idxs=idxs,
            tol=tol
        )
        
        return discs, radii
    
    except IndexError:
        print(
            'The model did not return any detections.'\
            ' Check to make sure the arguments starting_angle and increase_angle are set correctly.'\
            'You can also try changing the model, changing windowing method, turn PSD on/off, set tol to None, etc. to see if it helps.'
         )
        
        exit(0)


def get_reconstruction(tiff_path, params):
    # Read the tiff file
    imgs = read_tiff(tiff_path)

    if imgs[0].shape[0] != imgs[0].shape[1]:
        print('The input image is not square. The image will be cropped to be a*a where `a` is the smallest dimension.')
        a = min(imgs[0].shape[0], imgs[0].shape[1])
        slice_x, slice_y = slice(0, a), slice(0, a)
        imgs = [img[slice_x, slice_y] for img in imgs]
    
    imgs = [cp.array(img) for img in imgs]  # Transfer to GPU

    IMAGESIZE = imgs[0].shape[0]
    scale = params['scale']            
    hres_size = (IMAGESIZE * scale, IMAGESIZE * scale)

    del params['scale']

    # Reconstruction
    print('Performing Reconstruction...', end='')
    obj, pupil = reconstruct_v2(
        imgs,
        discs,
        row,
        hres_size,
        **params
    )
    print('Done!')
    return obj, pupil, imgs

        
def save_illumination(discs, radii, params):
    print('Saving illumination results...', end='')
    os.makedirs(params['illumination']['savedir'], exist_ok=True)

    if params['illumination']['format'].lower() == 'mat':
        mat_file = {
            'discs': discs,
            'radii': radii
        }

        savepath = os.path.join(
            params['illumination']['savedir'],
            os.path.basename(tiff_path) + '.mat'
        )

        savepath = unique_path(savepath)
        io.savemat(savepath, mat_file)
#         print(f'Illumination output saved to {savepath}')
        print('Done!')

    elif params['illumination']['format'].lower() == 'npz':
        discs = np.array(discs)
        radii = np.array(radii)

        savepath = os.path.join(
            params['illumination']['savedir'],
            os.path.basename(tiff_path) + '.npz'
        )

        savepath = unique_path(savepath)
        np.savez(savepath, discs=discs, radii=radii)
#         print(f'Illumination output saved to {savepath}')
        print('Done!')

    else:
        print(f"{params['illumination']['format']} format not recognised. Only mat and npz are supported.")


def save_reconstruction(obj, pupil, imgs, params):
    print('Saving reconstruction results...', end='')

    os.makedirs(params['reconstruction']['savedir'], exist_ok=True)
    savepath = os.path.join(
        params['reconstruction']['savedir'],
        os.path.basename(tiff_path)
    )

    if params['reconstruction']['format'].lower() == 'png':
        # Amp
        im = cp.asnumpy(to_uint8(cp.abs(obj)))
        cv2.imwrite(unique_path(savepath + '_amp.png'), im)
        
        # Phase
        im = cp.asnumpy(to_uint8(cp.angle(obj)))
        cv2.imwrite(unique_path(savepath + '_phase.png'), im)
        
        # Mean Image
        mean_img = cp.array(imgs).mean(axis=0)
        im = cp.asnumpy(to_uint8(mean_img))
        cv2.imwrite(unique_path(savepath + '_raw_mean.png'), im)

        # Pupil Amp
        im = cp.asnumpy(to_uint8(cp.abs(pupil)))
        cv2.imwrite(unique_path(savepath + '_pupil_amp.png'), im)
        
        # Pupil Phase
        im = cp.asnumpy(to_uint8(cp.angle(pupil)))
        cv2.imwrite(unique_path(savepath + '_pupil_phase.png'), im)
        print('Done!')
        
    elif params['reconstruction']['format'].lower() == 'tiff':
        # Amp
        skimage.io.imsave(
            unique_path(savepath + '_amp.tiff'),
            cp.asnumpy(cp.abs(obj))
        )
        
        # Phase
        skimage.io.imsave(
            unique_path(savepath + '_phase.tiff'),
            cp.asnumpy(cp.angle(obj))
        )
        
        # Mean Image
        mean_img = cp.array(imgs).mean(axis=0)
        skimage.io.imsave(
            unique_path(savepath + '_raw_mean.tiff'),
            cp.asnumpy(mean_img)
        )
        
        # Pupil Amp
        skimage.io.imsave(
            unique_path(savepath + '_pupil_amp.tiff'),
            cp.asnumpy(cp.abs(pupil))
        )
        
        # Pupil Phase
        skimage.io.imsave(
            unique_path(savepath + '_pupil_phase.tiff'),
            cp.asnumpy(cp.angle(pupil))
        )
        print('Done!')
    else:
        print(f"{params['reconstruction']['format']} format not recognised. Only png and tiff are supported.")

        
def unique_path(f):
    """
    Creates unique path.
    """
    i = 1
    name, ext = os.path.splitext(f)
    
    while os.path.exists(f):
        f = f'{name}_{i}{ext}'
        i += 1
    
    return f


if __name__ == '__main__':
    from sys import argv

    if len(argv) < 2:
        print(
            'Usage:\n\t python3 fpm.py /path/to/tiff/file \n'\
            'Specify parameters in params.py.'
        )
        exit(0)

    tiff_path = argv[1]

    # Illumination Estimation
    discs, radii = get_illumination(tiff_path, **illumination_params)

    # Save Illumination Estimation Results
    save_illumination(discs, radii, save_params)

    # Reconstruction
    obj, pupil, imgs = get_reconstruction(tiff_path, reconstruction_params)
    
    # Save Reconstruction Results
    save_reconstruction(obj, pupil, imgs, save_params)