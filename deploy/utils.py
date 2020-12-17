import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean as euclid
from scipy.io import loadmat
from tqdm.auto import tqdm
try:
    from detectron2.structures import BoxMode
    from detectron2.engine import DefaultPredictor
except ImportError:
    print('Reminder! Fix CUDA 10.1 error, Detectron2 cannot be used till then. Ignoring for now.')
from IPython.display import clear_output
import pickle
import json
import warnings


holdout_n_val = [
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abl.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abm.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abn.tif',
    'MAX_20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018aba.tif',
    'MAX_20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abb.tif',
    'MAX_20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abc.tif',
    'MAX_20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abd.tif',
    'MAX_20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abe.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018aba.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abb.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abc.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abd.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abe.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abf.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abg.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abh.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abi.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abj.tif',
    '20190717_H9c2-G1_mCh-OMP_inFBS_1520_convtTL_018abk.tif',
]


def equate_amp_phase(amp_, phase_):
    """
    equate_amp_phase(amp_, phase_)
        Creates amplitude-phase pairs, used internally for dataset creation.
        
    Parameters
    ----------
    `amp_`: String, required
            Path to the amplitude folder.
    `phase_`: String, required
            Path to the phase folder.
    """
    amps = glob.glob(os.path.join(amp_, '*')) 
    phases = glob.glob(os.path.join(phase_, '*')) 
    
    assert (len(amps)!=0) and (len(phases)!=0), "One or both of the folder(s) is empty."
    
    for f1, f2 in zip(amps, phases): 
        os.rename(f1, os.path.join(os.path.dirname(f1), os.path.basename(f2)))
    
    print('Done.')


def generate_labels_csv(images_path, save_path):
    """
    generate_labels_csv(images_path, save_path)
        Generates labels.csv file for  the dataset, used internally for dataset creation. 
    
    Parameters
    ----------
    `images_path`: String, required
            Path to the 'images' directory of the dataset. 
    `save_path`: String, required
            Path where to save the csv file (should end with .csv).
    
    """
    files = glob.glob(os.path.join(images_path, '*.tif')) + glob.glob(os.path.join(images_path, '*.tiff'))
    
    assert len(files)!=0, "The specified folder does not contain any TIFF files. Check the path and try again."

    d = {'file': []}

    dicts = [(f, loadmat(f.replace('.tiff', '.mat').replace('.tif', '.mat').replace('images', 'labels'))) for f in files]

    ignore_keys = ['__header__', '__version__', '__globals__']  # Not needed, hence ignored

    for f, item in dicts:
        # Filename
        d['file'].append(f.replace('.mat', '.tif'))
        
        for key, value in item.items():
            if key in ignore_keys:
                continue
            # Squeeze numpy arrays
            if isinstance(value, np.ndarray):
                value = value.squeeze().tolist()

            # Merge all other keys
            if key not in d:
                d[key] = []
            
            # Store to the dictionary
            d[key].append(value)

    df = pd.DataFrame(d)  # Convert to DataFrame
    df.to_csv(save_path, index=False)  # Save the CSV file
    print(f'{save_path} written succesfully.')
    

def read_tiff(img_path, uint8=True, cm=None):
    """
    read_tiff(img_path, uint8=True, cm=None)
        Reads a multi-page tiff file.
    
    Parameters
    ----------
    `img_path`: String, required
            Path to the tiff file.
    `uint8`: Boolean, optional
            Specify whether to downscale the image to uint8 or not. Defaults to True.
    `cm`: Integer (range: [1, 12]), optional
            Defaults to None. If specified, the corresponding colormap is applied
            and returned along with the raw images.
    
    Returns
    -------
    `imgs`: A list of images read from the tiff file.
    (conditional) `imgs_cm`: The colormap-applied images in a list. Returned only if `cm` is specified.
    """
    # Read the image
    read, imgs = cv2.imreadmulti(img_path, flags=cv2.IMREAD_ANYDEPTH) 
    assert read, "The image could not be read. Make sure the file is accessible and try again."
    
    if uint8:
        # Scale down to uint8
        imgs = [cv2.convertScaleAbs(img, alpha=(255/img.max())) for img in imgs] 
    if cm is not None: 
        # Apply colormap (for visualization only)
        imgs_cm = [cv2.applyColorMap(img, cm) for img in imgs] 
        return imgs, imgs_cm

    return imgs


def get_label(image_path, labels, return_row=False, return_delta=False):
    """
    get_label(image_path, labels, return_row=False, return_delta=False)
        Fetch the label (bounding box coordinates for discs) for a given tiff file.
    
    Parameters
    ----------
    `image_path`: String, required
            Path of the image. Needs to be as specified in the labels file.
    `labels`: Pandas DataFrame, required
            The labels dataframe.
    `return_row`: Boolean, optional
            If set to True, returns the complete label row. For testing. Defaults to False.
    `return_delta`: Boolean, optional
            If set to True, returns delta_x, delta_y and r. Defaults to False.
            
    Returns
    -------
    `disc1`, `disc2`:  Lists of (x, y, w, h) coordinates of the bounding box enclosing the main
            (disc1) as well as the mirror disc (disc2), for all the frames.
    (conditional) `row`: Pandas DataFrame containing the row from labels corresponding to the requested file.
            Returned only if `return_row` is True.
    (conditional) `delta_x`, `delta_y` and `r`: delta_x, delta_y and r corresponding to the requested file.
            Returned only if `return_delta` is True.
    """
    # Get the row corresponding to the required image
    row = labels[labels.file == image_path]
    assert len(row) != 0, "The image path could not be found in the labels file."
    
    if return_row:
        return row
    
    k0x = np.array(row.K0X.map(eval).values[0])
    k0y = np.array(row.K0Y.map(eval).values[0])
    NA = row.NA
    PIXELSIZE = int(row.PIXELSIZE)
    IMAGESIZE = int(row.IMAGESIZE)
    RI = float(row.RI)
    MAGNIFICATION = int(row.MAGNIFICATION)
    ILLUMINATION_OFFCENTER_X = float(row.ILLUMINATION_OFFCENTER_X)
    ILLUMINATION_OFFCENTER_Y = float(row.ILLUMINATION_OFFCENTER_Y)
    WAVELENGTH = float(row.WAVELENGTH)    
    
    CUTOFF_FREQ = 2 * np.pi * NA * RI / WAVELENGTH  # formula
    NYQUIST_FREQ = 2 * np.pi / (2 * PIXELSIZE / (RI * MAGNIFICATION))  # formula
    
    # Origin is at the center of the image, will need to be shifted
    orig = (IMAGESIZE/2) - 1  # -1, as the indexing starts from 0 in python
    
    # Coordinates of the center of the disc
    delta_x = (IMAGESIZE/2) * k0x / NYQUIST_FREQ + ILLUMINATION_OFFCENTER_X
    delta_y = (IMAGESIZE/2) * k0y / NYQUIST_FREQ + ILLUMINATION_OFFCENTER_Y
    
    # Radius of the disc
    r = float(IMAGESIZE * (CUTOFF_FREQ / NYQUIST_FREQ) / 2)
    
    if return_delta:
        return delta_x, delta_y, r
    
    # Shift origin to top-left corner to make it compatible with OpenCV
    delta_x = delta_x + orig
    delta_y = delta_y + orig
    
    # Calculate bounding box coordinates for main as well as the mirror disc
    # in XYWH format, XY are coordinate of the top left corner
    disc1 = [
        [max(float(x - r), 0) , max(float(y - r), 0), 2*r, 2*r] 
        for x, y in zip(delta_x, delta_y)
    ]
    
    disc2 = [
        [max(float(2 * orig - x - r), 0) , max(float(2 * orig - y - r), 0), 2*r, 2*r]
        for x, y in zip(delta_x, delta_y)
    ]
    return disc1, disc2


def preprocess(imgs, visualize=False, return_rgb=False, preprocess_fft=True, resize=None):
    """
    preprocess(imgs, visualize=false, return_rgb=False)
        Preprocess a list of images for training using the following pipeline. 
        Image > Magnitude Spectrum > NL Means Denoising > Bilateral Filter > Morph. Closing > Sharpening 

    Parameters
    ----------
    `imgs`: list or a Numpy array, required
            A single image or a list of images to be preprocessed.
    `visualize`: Boolean, optional
            Defaults to False. If set to True, visualizes the last processed image.
            For debugging/testing
    `return_rgb`: Boolean, optional
            Defaults to False. If set to True, triplicates the grayscale image into 3 channels. Required 
            while evaluating the model.
            
    Returns
    -------
    `out`: List of preprocessed images, in the order they were provided.
    """
    
    if not isinstance(imgs, list):  # A single image is provided
        imgs = [imgs,]  # To reduce redundant code later in the function
        
    # Calculate FFT
    fs = [20*np.log(1 + np.abs(np.fft.fft2(img))) for img in imgs]  # magnitude spectrum
    fshifts = [np.fft.fftshift(f) for f in fs]  # Shift zero frquency component
    fshifts = [cv2.convertScaleAbs(fshift, alpha =255/fshift.max()) for fshift in fshifts] # Convert to uint8

    # Pre-processing FFT images
    if preprocess_fft:  # See if it is asked for
        kernel_sharpening = np.array(
                [[0,-1,0],
                [-1,+5,-1],  # The sharpen kernel, required for sharpening
                [0,-1,0]]
        )

        out = []
        for fshift in fshifts:
            fshift1 = cv2.fastNlMeansDenoising(fshift, 9, 9, 7, 21)  # Denoising
            fshift2 = cv2.bilateralFilter(fshift1, 5, 75, 75)  # Bilateral Filter (Blurring)
            fshift2 = cv2.morphologyEx(fshift2, cv2.MORPH_CLOSE, np.ones((3, 3)))  # Closing
            fshift2 = cv2.filter2D(fshift2, -1, kernel_sharpening)  # Sharpening
            out.append(fshift2)  # Store the result
        
        # Visualize the results
        if visualize:
            plt.figure(figsize=(15, 45))
            plt.subplot(131)
            plt.title('Input')
            plt.imshow(fshift)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(132)
            plt.title('NL Means Denoising')
            plt.imshow(fshift1)
            plt.xticks([])
            plt.yticks([])
            plt.subplot(133)
            plt.title('Bilateral Filter + Closing + Sharpening')
            plt.imshow(fshift2)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.show()
    else:  # Return without preprocessing
        out = fshifts  

    if resize is not None:
        out = [cv2.resize(img, (resize, resize)) for img in out]
    
    if return_rgb:
        out = [np.dstack([im, im, im]) for im in out]  # Stack along 2nd axis to get (imagesize, imagesize, 3) shape

    return out


def generate_frames(tiff_path, save_path, preprocess_fft=True):
    """
    generate_frames(tiff_path, save_path, preprocess_fft=True)
        Generate and store individual frames from the tiff images.
        
    Parameters
    ----------
    `tiff_path`: String, required
            Path to the folder where the tiff files are stored.
    `save_path`: String, required
            Path to the folder where the frames need to be saved.
    `preprocess_fft`: Boolean, optional
            Whether to preprocess the magnitude spectrum or not. If set to False, 
            the magnitude spectrum does not undergo any preprocessing. Defaults to True.
    """
    os.makedirs(save_path, exist_ok=True)
    files = glob.glob(os.path.join(tiff_path, '*.tiff')) + glob.glob(os.path.join(tiff_path, '*.tif'))
    for f in tqdm(files, desc='Progress', total=len(files)):
        # Read and preprocess frames
        imgs = read_tiff(f)
        imgs = preprocess(imgs, preprocess_fft=preprocess_fft)

        # Save each of the frames individually
        for i, img in enumerate(imgs):
            ret = cv2.imwrite(os.path.join(save_path, f'{os.path.splitext(os.path.basename(f))[0]}_{i}.png'), imgs[i])

            if not ret:  # Image not written
                print(f"[W] Couldn't write {os.path.join(save_path, f'{os.path.splitext(os.path.basename(f))[0]}_{i}.png')}.")
                

def df_to_json(df, images_path, dropped_frames=[],out_file=None, overwrite=False):
    """
    df_to_json(df, images_path, dropped_frames=[], out_file=None, overwrite=False)
        Creates JSON annotations file in COCO format. Made for training with detectron2.
        
    Parameters
    ----------
    `df`: Pandas DataFrame, required
            The labels DataFrame for the dataset.
    `images_path`: String, required
            Path to the images folder where all the PNG images are stored (using generate_frames). Should be same as 
            `save_path` in generate_frames.
    `dropped_frames`: String or List, optional
            List of frames that were dropped using empty frame remover. The function will not warn you if
            these files do not exist. You can either pass a python list or path to pickle file containing the list.
            Defaults to [].
    `out_file`: String, optional
            The path where to save the JSON file. Defaults to None, in which case, the created dictionary is returned (used
            for testing/debugging).
    `overwrite`: Boolean, optional
            If `out_file` already exists, whether to overwrite it. Defaults to False. 
            If False, saves the file with a different name. If set to True, the file is overwritten.

    Returns
    -------
    (conditional) coco_dict: The dictionary created in COCO format. It is returned only if `out_file` 
    is set to None or is not provided. Used for testing/debugging.
    
    [Note] If `out_file` is set , nothing is returned by the function.
    """
    assert isinstance(df, pd.DataFrame), "[!] 'labels' must be a pandas DataFrame!"
    assert out_file is None or isinstance(out_file, str), "[!] 'out_file' must be a string!"
    
    frames = int(df.FRAMES[0])
    image_size = int(df.IMAGESIZE[0])
    
    # Load pickled dropped_frames list if the path is given
    if isinstance(dropped_frames, str):
        dropped_frames = pickle.load(open('removed_frames.pkl', 'rb'))

        
    assert isinstance(dropped_frames, list), "dropped_frames must be a list. You can also"\
        "pass path to a pickle file containing the dropped frames list"
    
    # Categories (only one here)
    categories= [ 
        {"supercategory": "Disc","id": 1,"name": "Disc"}
    ]
    
    # Images
#     print('Generating a list of images...')
    file_names=df["file"]
    
    images=[]
    for i, f in tqdm(enumerate(file_names), desc='Generating a list of images', total=len(file_names)):
        for j in range(frames):
            fname = os.path.join(images_path, f'{os.path.splitext(os.path.basename(f))[0]}_{j}.png')

            if fname in dropped_frames:
#                 print(f'[i] {fname} is in dropped frames list. Skipped.')
                continue
            elif (not os.path.exists(fname)):
                print(f'[W] {fname} does not exist and is not in dropped frames list. Skipped.')
                continue

            Dict={"file_name": fname,"height":image_size,"width":image_size,"id":i * frames + j + 1}
            images.append(Dict)
    
#     print('\nDone!\n')
    
    # Annotations
    
    # Get bounding box coordinates for both the discs
    labels = [get_label(f, df) for f in file_names]
    
#     print('Generating a list of annotations...')
    annotations=[]
    
    # Loop through tiff files
    for i, f in tqdm(enumerate(file_names), desc='Generating a list of annotations', total=len(file_names)):
        disc1, disc2 = labels[i]  # Label for frames in f
        
        for j in range(frames):
            fname = os.path.join(images_path, f'{os.path.splitext(os.path.basename(f))[0]}_{j}.png')
            
            if fname in dropped_frames:
#                 print(f'[i] {fname} is in dropped frames list. Skipped.')
                continue
            elif (not os.path.exists(fname)):
                print(f'[W] {fname} does not exist and is not in dropped frames list. Skipped.')
                continue

            main_disc = {
                "image_id": i * frames + j + 1,  # Same as used in the 'Images' section
                "bbox": disc1[j],
                "area": disc1[j][2] * disc1[j][3],
                "bbox_mode": BoxMode.XYWH_ABS,
                "iscrowd": 0,
                "category_id": 1,  # Hardcoded, for single category
                "id": i * 2 * frames + j*2 + 1,
            }

            mirror_disc = {
                "image_id": i * frames + j + 1,  # Same as used in the 'Images' section
                "bbox": disc2[j],
                "area": disc2[j][2] * disc2[j][3],
                "bbox_mode": BoxMode.XYWH_ABS,
                "iscrowd": 0,
                "category_id": 1,  # Hardcoded, for single category
                "id": i * 2 * frames + j*2 + 2,
            }
            annotations.append(main_disc)
            annotations.append(mirror_disc)

    print('\nDone!\n')
    
    # COCO JSON format
    coco_dict={"images":images,"categories":categories,"annotations":annotations}

    # Save JSON or return dict (for DEBUG)
    if out_file is not None:
        if not overwrite:
            name, ext = os.path.splitext(out_file)
            i = 1
            while os.path.exists(out_file):
                out_file = f'{name}_{i}.{ext}'
                i += 1

        print(f'Saving {out_file}...', end='')
        with open(out_file, 'w') as fp:
            json.dump(coco_dict, fp)
        
        print('Done!\n')
            
    else:
        return coco_dict

    
def remove_empty_frames_legacy(tiff_path, frames_path, last_file=None, delete=False):
    """
    remove_empty_frames_legacy(tiff_path, frames_path, last_file=None, delete=False)
        Please use the newer function remove_empty_frames. This is only kept for testing purpose.
        To be used for removal of empty frames from the dataset. NOTE: Designed to be used inside IPython Notebook only.
    
    Parameters
    ----------
    `tiff_path`: String, required
            Path to the directory containing tiff files.
    `frames_path`: String, required
            Path to the directory where the frames are saved.
    `delete`: Boolean, optional
            Whether to delete the selected files right away. Defaults to False.
    `last_file`: String, optional
            If the process was interrupted previously, the function stores a backup
            that can be loaded later. To skip re-doing all the frames already done, 
            you can directly specify a filename here and the script will start from there.
            Defaults to None.

    Returns
    -------
    `to_del`: List of frames to be deleted is returned back for later use.
    
    """
    to_del = []
    logs = []
    files = glob.glob(os.path.join(tiff_path, '*.tiff')) + glob.glob(os.path.join(tiff_path, '*.tif'))
    files = sorted(files)
    
    if last_file is not None:
        if last_file in files:
            idx = files.index(last_file)
        else:
            logs.append(f'{last_file} could not be found. Starting from the beginning.')
            idx = 0
    else:
        idx = 0

    f_prev = None   # Used to show a message at the top.
    for k, f in tqdm(enumerate(files[idx:]), desc='Progress', total=len(files)):
        if f_prev is not None:  # Print frames added last.
            print(f'Frame numbers {nums} of {f_prev} added to removal list.')

        if len(logs) > 0:  # Print logs if any
            print('*'*40)
            print('Logs')
            for log in logs:
                print(log)
            print('*'*40)

        print('*'*40)
        print(' '*10, 'Empty Frame Remover (manual)', ' '*10)
        print('Use this tool to remove empty frames.\n'
              'Type in the frame numbers you want to get rid of, separated by a space (\' \').\n'
              'Enter q to exit.'
             )
        print('*'*40)
        print(f'{k + 1}/{len(files)}. {f}', end='\n\n')

        # Read and preprocess the tiff files
        imgs = read_tiff(f)
        imgs = preprocess(imgs, visualize=False)

        # Show frames in a 5*12 grid
        fig, axes = plt.subplots(
            nrows=5, ncols=12, sharex=True, sharey=True
        )

        for i in range(5):
            for j in range(12):
                ax = axes[i, j]
                img = imgs[i*12 + j]
                ax.imshow(img)
                ax.set_title(f'{i*12 + j}')

        plt.show()

        # Take frame numbers as input from the user
        text=input()
        
        if text == 'q':  # Exit condition
            print('Exiting. No changes were made. Returning list so far...')
            
            if len(to_del) > 0:
                with open('./to_del_backup.txt', 'a+') as f:
                    f.write(f'\n\nLast file: {f}\n')
                with open('./to_del_backup.txt', 'a') as f:
                    f.write('\n'.join(to_del))
            return to_del

        if text == '':  # Ignore Empty
            nums = []
            f_prev = f
            clear_output()  # Clear the output screen
            continue

        try:
            nums = list(map(int, text.strip().split(' ')))
            fnames = [os.path.join(frames_path, f'{os.path.splitext(os.path.basename(f))[0]}_{i}.png') for i in nums]  # Naming format
            to_del.extend(fnames)  # Add to the removal list
        
        except Exception as e:
            logs.append(f'{e} encountered while processing {f}. Input was {text}')
            
            if len(to_del) > 0:
                with open('./to_del_backup.txt', 'a+') as f:
                    f.write(f'\n\nLast file: {f}\n')
                with open('./to_del_backup.txt', 'a') as f:
                    f.write(to_del.join('\n'))
            
            
            
        clear_output()  # Clear the output screen
        f_prev = f  # Used to show a message at the top.
    
    print(f'Done! A total of {len(to_del)} frames are to be removed.', end=' ')
    if delete:
        print('Deleting frames...', end='')
        for f in tqdm(to_del, desc='Deleting frames...', total=len(to_del)):
            try:
                os.remove(f)
            except FileNotFoundError:
                print(f'{e} encountered while processing {f}.')
                
        print('Done!')

    return to_del


def _chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def remove_empty_frames(tiff_path, frames_path, last_file=None, delete=False, min_mean=100, max_mean=105):
    """
    remove_empty_frames(tiff_path, frames_path, last_file=None, delete=False, min_mean=100, max_mean=105)
        To be used for removal of empty frames from the dataset. 
        NOTE: Designed to be used inside IPython Notebook only.
    
    Parameters
    ----------
    `tiff_path`: String, required
            Path to the directory containing tiff files.
    `frames_path`: String, required
            Path to the directory where the frames are saved.
    `delete`: Boolean, optional
            Whether to delete the selected files right away. Defaults to False.
    `last_file`: String, optional
            If the process was interrupted previously, the function stores a backup
            that can be loaded later. To skip re-doing all the frames already done, 
            you can directly specify a filename here and the script will start from there.
            Defaults to None.
    `min_mean`, `max_mean`: Number (Integer/Float), optional
            - If a frame has mean < min_mean, it is added to removal list.
            - If a frame has mean > max_mean, it is not added to removal list.
            - If a frame has min_mean < mean < max_mean, it is marked suspicious and
              The user has to manually select empty frames from them.
            `min_mean` defaults to 100, `max_mean` defaults to 105.

    Returns
    -------
    `to_del`: List of frames to be deleted is returned back for later use.
    """
    to_del = []
    logs = []
    suspects = []
    
    files = glob.glob(os.path.join(tiff_path, '*.tiff')) + glob.glob(os.path.join(tiff_path, '*.tif'))
    files = sorted(files)

    for k, f in tqdm(enumerate(files), desc='Recognizing empty frames...', total=len(files)):
        # Read and preprocess the tiff files
        imgs = read_tiff(f)
        imgs = preprocess(imgs, preprocess_fft=False, visualize=False)
        
        # Compute mean to determine if the frame is empty
        for i, img in enumerate(imgs):
            if img.mean() > max_mean:  # mean > max_mean
                continue
            elif img.mean() < min_mean:  # mean < min_mean
                to_del.append(
                    os.path.join(frames_path, f'{os.path.splitext(os.path.basename(f))[0]}_{i}.png')
                )
            else:  # min_mean < mean < max_mean
                suspects.append((f, i))

    # Add a message at the top
    logs.append(f'{len(to_del)} frames automatically added to the removal list.')
    
    # Split in parts (for displaying)
    grid_h = 5  # Height  of the grid
    grid_w = 12  # Width of the grid
    n_chunks = np.ceil(len(suspects) / (grid_h * grid_w))
    chunks = _chunks(suspects, grid_h * grid_w)
    
    # Iterate over chunks and display them to the user
    for number, chunk in enumerate(chunks):
        if len(logs) > 0:  # Print logs if any
            print('*'*40)
            print('Logs')
            for log in logs:
                print(log)
            print('*'*40)

        print('*'*40)
        print(' '*10, 'Empty Frame Remover (semi-automatic)', ' '*10)
        print('Use this tool to remove empty frames.\n'
              'This tool will automatically pick highly probable frames to be removed. Some suspicious frames will be shown (~12% of the total).\n'
              'From the suspicious frames, type in the frame numbers you want to get rid of, separated by a space (\' \').\n'
              'Enter q to exit.'
             )
        print('*'*40)
        pbar = tqdm(desc=f'Processing chunk {number + 1} / {int(n_chunks)}', total=grid_h * grid_w)

        # Show frames in a 4*12 grid
        if len(chunk) == grid_h * grid_w:
            fig, axes = plt.subplots(
                nrows=grid_h, ncols=grid_w, sharex=True, sharey=True
            )
            fig.set_size_inches(1.25*grid_w, 1.25*grid_h)
            
            # Plot images in the grid
            for i in range(grid_h):
                for j in range(grid_w):
                    ax = axes[i, j]
                    f, idx = chunk[i * grid_w + j]
                    
                    # print(f, idx)  # DEBUG
                    
                    # Read and preprocess the image
                    img = read_tiff(f)[idx]
                    img = preprocess(img, preprocess_fft=True, visualize=False)[0]
                    ax.imshow(img)
                    ax.set_title(f'{i * grid_w + j}')
                    pbar.update(1)
            plt.show()
            
        else:
            print(f'< {grid_h * grid_w} frames are remaining. Showing all at once.')
            for i in range(len(chunk)):
                f, idx = chunk[i]

                # Read and preprocess the tiff files
                img = read_tiff(f)[idx]
                img = preprocess(img, preprocess_fft=True, visualize=False)[0]

                plt.imshow(img)
                plt.title(f'{i}')
                plt.show()

        # Take frame numbers as input from the user
        text=input()

        if text == 'q':  # Exit condition
            print('Exiting. No changes were made. Returning list so far...')

            if len(to_del) > 0:
                with open('./to_del_backup.txt', 'a+') as f:
                    f.write(f'\n\nLast file: {f}\n')
                with open('./to_del_backup.txt', 'a') as f:
                    f.write('\n'.join(to_del))
            return to_del

        if text == '':  # Ignore Empty
            nums = []
            clear_output()  # Clear the output screen
            continue

        try:
            nums = list(map(int, text.strip().split(' ')))

            # Quite complex. chunk[i][0] -> filename, chunk[i][1] -> frame no.
            fnames = [
                os.path.join(
                    frames_path,
                    f'{os.path.splitext(os.path.basename(chunk[i][0]))[0]}_{chunk[i][1]}.png'  # Naming Format
                ) for i in nums
            ]

            to_del.extend(fnames)  # Add to the removal list

        # Log errors encountered, if any
        except Exception as e:
            logs.append(f'{e} encountered while processing {f}. Input was {text}')

            # Backup the list so far
            if len(to_del) > 0:
                with open('./to_del_backup.txt', 'a+') as f:
                    f.write(f'\n\nLast file: {f}\n')
                with open('./to_del_backup.txt', 'a') as f:
                    f.write(to_del.join('\n'))

        clear_output()  # Clear the output screen
    
    print(f'Done! A total of {len(to_del)} frames are to be removed.', end=' ')

    if delete:
        print('Deleting frames...', end='')
        for f in tqdm(to_del, desc='Deleting frames...', total=len(to_del)):
            try:
                os.remove(f)
            except FileNotFoundError:
                print(f'{e} encountered while processing {f}.')
                
        print('Done!')

    return to_del


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))


def _get_r(discs, orig):
    rs = []
    for disc in discs:
        if disc==0:
            rs.append(0)
        else:
            x, y = disc[0]
            x, y = abs(x - orig), abs(y - orig)
            rs.append(np.sqrt(x**2 + y**2))
    return rs
            


def _get_discs(preds, imagesize, idxs=None, total=None, calibrate=False, calibrate_v2=False, tol=None, fill_empty=False, return_discs_raw=False):
    """
    _get_discs(preds, imagesize)
        Used internally by Predictor class. Automatically identifies the disc to be returned.
        Assumes the frames start from theta = 0 degrees (origin at the center of the image).
        
    Parameters
    ----------
    `preds`: List, required
            The list of predictions returned by the model.
    `imagesize`: Integer, required
            Length of one side of the image. The image is assumed to have
            same length and width.
            
    Returns
    -------
    `discs_final`: List of shortlisted discs' center coordinates (x, y). If no discs were found in a frame,
            0 is put at that index.
    `radii`: List of radii of shortlisted discs. If no discs were found in a frame, 0 is put at that index.
    """
    discs = []
    radii = []
    
    orig = imagesize // 2 - 1  # Origin
    for i, pred in enumerate(preds):
        if len(pred) == 0:  # Avoid empty predictions
            discs.append(0)
            radii.append(0)
            continue
            
        # Get center of the disc
        x1 = pred[:, 0]  # Top Left Corner X
        y1 = pred[:, 1]  # Top Left Corner Y
        x2 = pred[:, 2]  # Bottom Right Corner X
        y2 = pred[:, 3]  # Bottom Right Corner Y

        x = np.abs((x1 + x2) / 2 - orig).mean()
        y = np.abs((y1 + y2) / 2 - orig).mean()

        # Get both the discs' coordinates
        sign_x = (np.sign((x1 + x2) / 2 - orig)).astype(int)[0]
        sign_y = (np.sign((y1 + y2) / 2 - orig)).astype(int)[0]
        discs.append([
            [orig + sign_x * x, orig + sign_y * y],  # One of the discs
            [orig - sign_x * x, orig - sign_y * y]   # The disc opposite to it
        ])
        
        # Get radius of the disc
        radii.append((np.abs((y1 - y2) / 2).mean() + np.abs((x1 - x2) / 2).mean()) / 2)
        
    # Divide plane (360) into `frames` no. of divisions
    frames = len(preds) if total is None else total
    
#     r = np.mean(radii) # imagesize // 2 - 1  # Length of the vector
#     rs = _get_r(discs, orig)
    
#     thetas = list(range(90, 360, 360 // frames)) + list(range(0, 90, 360 // frames))  # 0:(360 // frames):360
#     thetas = list(np.mod(np.arange(90 ,  450, 360/frames), 360))
    thetas = list(np.arange(0 ,  360, 360/frames))
#     thetas = [np.deg2rad(t) for t in thetas] # DEBUG 28-05

#     x = lambda r, t: int(orig + r * np.cos(np.deg2rad(t)))
#     y = lambda r, t: int(orig + r * np.sin(np.deg2rad(t)))

#     points = [(x(r, theta), y(r, theta)) for r, theta in zip(rs, thetas)]
    ref = [1, 0]
    
    # Empty frames
    empty_frames = [True if disc==0 else False for disc in discs]
    
    # Choose the required disc out of the two discs
    discs_final = []
    if idxs is None:
        idxs = list(range(len(discs)))
    if tol is None or tol.lower() == 'none':
        tol = 360
    elif tol == 'auto':
        tol = 2 * 360/frames

    # Keep track of offsets
    offsets = []
    
    for idx, disc in zip(idxs, discs):
        if disc == 0:  # Avoid empty predictions
            discs_final.append(0)
            offsets.append([0, 0, 0, 0])
            continue

#         d0 = euclid(points[idx], disc[0])
#         d1 = euclid(points[idx], disc[1]) 

#         d0 = [disc[0][0] - orig, -(disc[0][1] - orig)]  # DEBUG 28-05
#         d0 = dot(d0, ref) / (dot(d0, d0) * dot(ref, ref))  # DEBUG 28-05
#         if np.sign(d0) == -1:  # DEBUG 28-05
#             d0 = 2*np.pi + d0  # DEBUG 28-05
#         d0 = np.mod(d0, 2*np.pi)  # DEBUG 28-05

        d0 = [disc[0][0] - orig, orig - disc[0][1]]  # DEBUG 28-05
        d0 = angle_between(ref, d0)  # DEBUG 28-05
        
        if disc == discs[-1]:  # Last disc might overshoot reference line
            off_d0_ = thetas[idx] - (360-d0)  # Offset without changing the sign
            d0_ = abs(thetas[idx] - (360-d0))
        
        off_d0 = thetas[idx] - d0
        d0 = abs(thetas[idx] - d0)
        
#         d1 = [disc[1][0] - orig, -(disc[1][1] - orig)]
#         d1 = dot(d1, ref) / (dot(d1, d1) * dot(ref, ref))
#         if np.sign(d1) == -1:
#             d1 = 2*np.pi + d1
#         d1 = np.mod(d1, 2*np.pi)
        d1 = [disc[1][0] - orig, orig - disc[1][1]]  # DEBUG 28-05
        d1 = angle_between(ref, d1)  # DEBUG 28-05
        if disc == discs[-1]:  # Last disc might overshoot reference line
            off_d1_ = thetas[idx] - (360-d1)
            d1_ = abs(thetas[idx] - (360-d1))
        off_d1 = thetas[idx] - d1
        d1 = abs(thetas[idx] - d1)
        
#         print(d0, d1)  #DEBUG
        if (d0 > tol) and (d1 > tol):
            discs_final.append(0)
            offsets.append([0, 0, 0, 0])
            continue

        if disc == discs[-1]:  # Last disc might overshoot reference line
            m = min(d0, d0_, d1, d1_)
            discs_final.append(disc[0] if (d0 == m) or (d0_ == m) else disc[1])

#             if m == d0:
#                 offsets.append(off_d0)
#             elif m == d0_:
#                 offsets.append(off_d0_)
#             elif m == d1:
#                 offsets.append(off_d1)
#             elif m == d1_:
#                 offsets.append(off_d1_)
            
            offsets.append([off_d0, off_d0_, off_d1, off_d1_])
            continue

        # Save correct offset (with sign)
        discs_final.append(disc[0] if d0 < d1 else disc[1])
        
        m = min(d0, d1)
        # Save correct offset (with sign)
#         if m == d0:
#                 offsets.append(off_d0)    
#         elif m == d1:
#                 offsets.append(off_d1)
        offsets.append([off_d0, np.inf, off_d1, np.inf])

#     print(tol, offsets)
    if calibrate:
        discs_final, radii = _calibrate(discs_final, radii, orig, idxs, total, fill_empty=fill_empty)
    elif calibrate_v2:
        discs_final, radii = _calibrate_v2(discs_final, radii, empty_frames, orig, idxs, total, fill_empty=fill_empty)
        
    if not return_discs_raw:
        return discs_final, radii
    else:
        return discs_final, radii, discs


def _calibrate_v2(discs, radii, empty_frames, orig, idxs=None, total=None, fill_empty=False):
    """TODO. Currently same as _calibrate"""
    nonzero = np.array([disc for disc in discs if disc != 0])
    del_x = abs(nonzero - orig)[:, 0]
    del_y = abs(nonzero - orig)[:, 1]
    r = (del_x ** 2 + del_y ** 2) ** 0.5
    r = r.mean(axis=0)
    
    frames = len(discs) if total is None else total
    if idxs is None:
        idxs = list(range(len(discs)))
    
    # Estitmate Offset
    ref = [1, 0]
    thetas_ = [angle_between()]
    
    thetas = list(np.arange(0, 360, 360/frames))  # Angle with +x axis
    
    x = lambda r, t: orig + r * np.cos(np.deg2rad(t))
    y = lambda r, t: orig + r * np.sin(np.deg2rad(t))
    
    if not fill_empty:
        discs_final = [[x(r, thetas[i]), y(r, thetas[i])] if disc != 0 else 0 for i, disc in zip(idxs, discs)]
    else:
        discs_final = [[x(r, thetas[i]), y(r, thetas[i])] for i, disc in zip(idxs, discs)]
        
    r = np.array([rad for rad in radii if rad != 0]).mean()
    radii_final = [r for _ in radii]
    return discs_final, radii_final
    

def _calibrate(discs, radii, orig, idxs=None, total=None, fill_empty=False):
    nonzero = np.array([disc for disc in discs if disc != 0])
    del_x = abs(nonzero - orig)[:, 0]
    del_y = abs(nonzero - orig)[:, 1]
    r = (del_x ** 2 + del_y ** 2) ** 0.5
    r = r.mean(axis=0)
    
    frames = len(discs) if total is None else total
    if idxs is None:
        idxs = list(range(len(discs)))

    
    thetas = list(np.arange(0, 360, 360/frames))  # Angle with +x axis
    x = lambda r, t: orig + r * np.cos(np.deg2rad(t))
    y = lambda r, t: orig + r * np.sin(np.deg2rad(t))
    
    if not fill_empty:
        discs_final = [[x(r, thetas[i]), y(r, thetas[i])] if disc != 0 else 0 for i, disc in zip(idxs, discs)]
    else:
        discs_final = [[x(r, thetas[i]), y(r, thetas[i])] for i, disc in zip(idxs, discs)]
        
    r = np.array([rad for rad in radii if rad != 0]).mean()
    radii_final = [r for _ in radii]
    return discs_final, radii_final


def fivenum(data):
    """Five-number summary."""
    return np.percentile(data, [0, 25, 50, 75, 100], interpolation='midpoint')


try:
    class Predictor(DefaultPredictor):
        def __init__(self, cfg, preprocess_fft=True):
            cfg.MODEL.RPN.BBOX_REG_LOSS_WEIGHT = 1.0
            cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
            cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 1.0
            cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "smooth_l1"
            
            super(Predictor, self).__init__(cfg)
            self.preprocess_fft = preprocess_fft

        def get_disc(self, tiff_path, visualize=False, warnings=True, return_boxes=False, return_discs_raw=False, idxs=None, total=None, slice_x=None, slice_y=None, calibrate=True, tol=None, fill_empty=False, _no_tqdm=False):
            """
            get_disc(tiff_path, visualize=False, return_boxes=False, _no_tqdm=False)
                Predict the center and radii of discs for the given tiff file.

            Parameters
            ----------
            `tiff_path`: Str, Required
                    Path to the tiff file.
            `visualize`: Boolean, optional
                    Defaults to False. If set to True, visualizes the predicted discs
                    on the given tiff file.
            `return_boxes`: Boolean, optional
                    Defaults to False. If set to True, returns the predicted bounding boxes in a list.
                    Used for testing/debugging.
            `_no_tqdm`: Boolean optional
                    Used internally. When set to True, does not display tqdm progress bar. Defaults to False.

            Returns
            -------
            `discs`: List of shortlisted discs' center coordinates (x, y). If no discs were found in a frame,
                0 is put at that index. 
            `radii`: Estimated radii of the discs. If no discs were found in a frame, 0 is put at that index.
            """
            assert (slice_x is None and slice_y is None) or (slice_x is not None and slice_y is not None), \
                "Provide either both or None of slice_x and slice_y"

            if isinstance(tiff_path, list):  # If a list of tiff files/numpy arrays is provided
                if isinstance(tiff_path[0], np.ndarray):
                    imgs = tiff_path
                else:
                    imgs = [read_tiff(t)[0] for t in tiff_path]

            elif isinstance(tiff_path, np.ndarray):  # If a numpy array is provided
                imgs = [tiff_path]

            else:  # If a single single/multipage tiff is provided
                imgs = read_tiff(tiff_path) 

            assert idxs is None or len(idxs) == len(imgs), "Provide enough indices for all of the images."

            # Preprocessing
            if slice_x is not None:  # If the image is to be cropped before processing
                imgs = [img[slice_x, slice_y] for img in imgs]

            imgs = preprocess(imgs, return_rgb=True, preprocess_fft=self.preprocess_fft)

            # Loop for all the frames in the tiff file.
            preds = []
            imgs_list = tqdm(enumerate(imgs), desc='Running Inference', total=len(imgs)) if not _no_tqdm else enumerate(imgs)
            for i, img in imgs_list:
                # print(img.shape)  # DEBUG: Should be (128, 128, 3)
                pred = self.__call__(original_image=img)['instances'].pred_boxes.tensor.cpu().numpy()
                if len(pred) > 2:  # Cannot be more than two discs. Rest must be erroneous.
                    if warnings:
                        print(f'[W] More than 2 detections found in {i}th frame.'
                               'Considering only the two most confident detections...'
                             )  # Print a warning
                    pred = pred[:2]  # Take the first 2 (they are sorted by confidence score)
                preds.append(pred)

            # Get Image Size
            imagesize = imgs[0].shape[0]

            # Identify the disc
            if not return_discs_raw:
                discs, radii = _get_discs(preds=preds, imagesize=imagesize, idxs=idxs, total=total, calibrate=calibrate, tol=tol, fill_empty=fill_empty)
            else:
                discs, radii, discs_raw = _get_discs(preds=preds, imagesize=imagesize, idxs=idxs, total=total, calibrate=calibrate, tol=tol, fill_empty=fill_empty, return_discs_raw=True)

            # Visualize Predictions
            if visualize and (len(imgs) != 60):
                print('The visualization code is only implemented for no. of frames = 60. Showing as much as possible ...')

            # Show frames in a 5*12 grid
            if visualize:
                print('Visualizing predictions...')
                if len(imgs) == 1:  # Show the image directly if only one is there
                    img = imgs[0]  # Grab the frame
                    d = discs[0]   # Grab the disc
                    r = radii[0]          # Grab the radius

                    mask = img.copy()  # Create mask

                    # Draw the discs on the mask
                    if d != 0 and r != 0:  # Avoid empty predictions
                        mask = cv2.circle(  # First disk
                            mask,
                            (int(d[0]), int(d[1])),
                            int(r),
                            (0, 0, 255),
                            -1
                        )
                    plt.imshow(cv2.addWeighted(img, 0.6, mask, 0.4, 1))  # Show the frame

                else:
                    fig, axes = plt.subplots(
                        nrows=5, ncols=12, sharex=True, sharey=True
                    )
                    fig.set_figheight(10)
                    fig.set_figwidth(24)

                    pbar = tqdm(total=len(imgs), desc='Drawing Circles')  # Progress bar
                    for i in range(5):
                        for j in range(12):
                            ax = axes[i, j]
                            try:
                                img = imgs[i*12 + j]  # Grab the frame
                                d = discs[i*12 + j]   # Grab the disc
                                r = radii[i]          # Grab the radius

                            except IndexError:  # imgs has < 60 frames:
                                continue  # Skip

                            mask = img.copy()  # Empty mask

                            # Draw the discs on the mask
                            if d != 0 and r != 0:  # Avoid empty predictions
                                mask = cv2.circle(  # First disk
                                    mask,
                                    (int(d[0]), int(d[1])),
                                    int(r),
                                    (0, 0, 255),
                                    -1
                                )

                            # Overlay mask on top of the frame and show
                            ax.imshow(cv2.addWeighted(img, 0.6, mask, 0.4, 1))  # Show the frame
                            pbar.update(1)  # Update Progress Bar
                    pbar.close()  # Stop the progress bar
                plt.tight_layout()
                plt.show()

            ret = [discs, radii]

            # Return predictions if asked
            if return_boxes:
                ret.append(preds)
            if return_discs_raw:
                ret.append(discs_raw)
            return ret

        def get_k(self, tiff_path, metadata, warnings=True):
            """
            get_k(tiff_path, metadata)
                Get *Real* (not fourier domain) K0X and K0Y values for an input tiff file.

            Parameters
            ----------
            `tiff_path`: String, required
                    Path to the tiff file.
            `metadata`: Dictionary, required
                    It should have these keys: PIXELSIZE, RI, MAGNIFICATION, IMAGESIZE,
                    ILLUMINATION_OFFCENTER_X, ILLUMINATION_OFFCENTER_Y with their values.

            Returns
            -------
            `k0x`, `k0y`: *Real-domain* K0X and K0Y values in a Numpy array.
            """
            PIXELSIZE = int(metadata['PIXELSIZE'])
            IMAGESIZE = int(metadata['IMAGESIZE'])
            RI = float(metadata['RI'])
            MAGNIFICATION = int(metadata['MAGNIFICATION'])
            ILLUMINATION_OFFCENTER_X = float(metadata['ILLUMINATION_OFFCENTER_X'])
            ILLUMINATION_OFFCENTER_Y = float(metadata['ILLUMINATION_OFFCENTER_Y'])

            NYQUIST_FREQ = 2 * np.pi / (2 * PIXELSIZE / (RI * MAGNIFICATION))  # formula
            discs, _ = self.get_disc(tiff_path, warnings=warnings)
            delta_x = np.array([disc if disc!=0 else [0, 0] for disc in discs])[:, 0]
            delta_y = np.array([disc if disc!=0 else [0, 0] for disc in discs])[:, 1]

            k0x = (np.array(delta_x) - ILLUMINATION_OFFCENTER_X) * NYQUIST_FREQ / (IMAGESIZE/2)
            k0y = (np.array(delta_y) - ILLUMINATION_OFFCENTER_Y) * NYQUIST_FREQ / (IMAGESIZE/2)
            return k0x, k0y

        def eval_disc_MAE(self, holdout_df, calibrate=False, fill_empty=True, print_fivenum=False, warnings=True):
            """
            eval_disc_MAE(holdout_df)
                Calculates MAE (Mean Absolute Error) for delta_x, delta_y and r.

           Parameters
            ----------
            `holdout_df`: Pandas DataFrame, required
                    Part of the labels dataframe you want to evaluate the model's performance on.

            Returns
            -------
            `mae_delta_x`, `mae_delta_y`, `mae_r`: MAE for delta_x, delta_y and r respectively.
            """
            assert isinstance(holdout_df, pd.DataFrame), 'Please pass a pandas DataFrame'

            # Get a list of all the files
            files = holdout_df.file
            mae_delta_x = {
                'px': [],  # in pixels
                'um': [],  # in micrometers
            }
            mae_delta_y = {
                'px': [],  # in pixels
                'um': [],  # in micrometers
            }
            mae_r = {
                'px': [],  # in pixels
                'um': [],  # in micrometers
            }
            mae_delta = {
                'px': [],  # in pixels
                'um': [],  # in micrometers
            }

            # Calculate MAE in a loop
            for file in tqdm(files, desc='Files completed', total=len(files)):
                # Get predictions
                discs, r = self.get_disc(
                    tiff_path=file,
                    visualize=False,
                    _no_tqdm=True, 
                    calibrate=calibrate,
                    fill_empty=fill_empty,
                    warnings=warnings
                )
                # Get the labels' row
                row = holdout_df[holdout_df.file==file]

                orig = int(row.IMAGESIZE) // 2 - 1

                delta_x = np.array([disc if disc!=0 else [orig, orig] for disc in discs])[:, 0] - orig
                delta_y = np.array([disc if disc!=0 else [orig, orig] for disc in discs])[:, 1] - orig
                
                delta = np.sqrt(delta_x ** 2 + delta_y ** 2)
                
                #Ignore zero values (corresponding to empty frames)
    #             nonzero = (np.array(delta_x) != 0).squeeze() & (np.array(delta_y) != 0).squeeze() & (np.array(r) != 0).squeeze()
                nonzero = np.invert((np.array(delta_x) == 0).squeeze() & (np.array(delta_y) == 0).squeeze()).squeeze()
                # Get ground truth
                delta_x_true, delta_y_true, r_true = get_label(file, holdout_df, return_delta=True)
                
                delta_true = np.sqrt(delta_x_true ** 2 + delta_y_true ** 2)


                # Calculate conversion factor
                PIXELSIZE = int(row.PIXELSIZE)
                IMAGESIZE = int(row.IMAGESIZE)
                RI = float(row.RI)
                MAGNIFICATION = int(row.MAGNIFICATION)
                NYQUIST_FREQ = 2 * np.pi / (2 * PIXELSIZE / (RI * MAGNIFICATION)) # formula
                conv_factor = 2 * NYQUIST_FREQ / IMAGESIZE

                # MAE calculation (micrometers)
                mae_delta_x['um'].append(
                    conv_factor * abs(((np.array(delta_x).squeeze()[nonzero]) - (delta_x_true[nonzero]))).mean()
                )
                mae_delta_y['um'].append(
                    conv_factor * abs(((np.array(delta_y).squeeze()[nonzero]) - (delta_y_true[nonzero]))).mean()
                )
                mae_delta['um'].append(
                    conv_factor * abs(((np.array(delta).squeeze()[nonzero]) - (delta_true[nonzero]))).mean()
                )
                mae_r['um'].append(
                    conv_factor * abs(np.array(r).squeeze()[nonzero] - r_true).mean()
                )

                # MAE calculation (pixels)
                mae_delta_x['px'].append(
                    abs(((np.array(delta_x).squeeze()[nonzero]) - (delta_x_true[nonzero]))).mean()
                )
                mae_delta_y['px'].append(
                    abs(((np.array(delta_y).squeeze()[nonzero]) - (delta_y_true[nonzero]))).mean()
                )
                mae_delta['px'].append(
                    abs(((np.array(delta).squeeze()[nonzero]) - (delta_true[nonzero]))).mean()
                )
                mae_r['px'].append(
                    abs(np.array(r).squeeze()[nonzero] - r_true).mean()
                )


                ##############
    #             # MAE calculation (micrometers)
    #             mae_delta_x['um'].append(
    #                 conv_factor * abs((abs(np.array(delta_x).squeeze()[nonzero]) - abs(delta_x_true[nonzero]))).mean()
    #             )
    #             mae_delta_y['um'].append(
    #                 conv_factor * abs((abs(np.array(delta_y).squeeze()[nonzero]) - abs(delta_y_true[nonzero]))).mean()
    #             )
    #             mae_r['um'].append(
    #                 conv_factor * abs(np.array(r).squeeze()[nonzero] - r_true).mean()
    #             )

    #             # MAE calculation (pixels)
    #             mae_delta_x['px'].append(
    #                 abs((abs(np.array(delta_x).squeeze()[nonzero]) - abs(delta_x_true[nonzero]))).mean()
    #             )
    #             mae_delta_y['px'].append(
    #                 abs((abs(np.array(delta_y).squeeze()[nonzero]) - abs(delta_y_true[nonzero]))).mean()
    #             )
    #             mae_r['px'].append(
    #                 abs(np.array(r).squeeze()[nonzero] - r_true).mean()
    #             )




            print(f'Mean absolute error in delta_x: {np.array(mae_delta_x["um"]).mean()} micrometers'
                  f'\t or \t {np.array(mae_delta_x["px"]).mean()} pixels')
            print(f'Mean absolute error in delta_y: {np.array(mae_delta_y["um"]).mean()} micrometers'
                  f'\t or \t {np.array(mae_delta_y["px"]).mean()} pixels')
            print(f'Mean absolute error in delta: {np.array(mae_delta["um"]).mean()} micrometers'
                  f'\t or \t {np.array(mae_delta["px"]).mean()} pixels')
            print(f'Mean absolute error in r: {np.array(mae_r["um"]).mean()} micrometers'
                  f'\t or \t {np.array(mae_r["px"]).mean()} pixels')
            
            if print_fivenum:
                print(f'fivenum of error in delta_x: \n {fivenum(np.array(mae_delta_x["um"]))} micrometers'
                      f'\n\t or \t {fivenum(np.array(mae_delta_x["px"]))} pixels')
                print(f'fivenum of error in delta_y: {fivenum(np.array(mae_delta_y["um"]))} micrometers'
                      f'\t or \t {fivenum(np.array(mae_delta_y["px"]))} pixels')
                print(f'fivenum of error in delta: {fivenum(np.array(mae_delta["um"]))} micrometers'
                      f'\t or \t {fivenum(np.array(mae_delta["px"]))} pixels')
                print(f'fivenum of error in r: {fivenum(np.array(mae_r["um"]))} micrometers'
                      f'\t or \t {fivenum(np.array(mae_r["px"]))} pixels')

            return mae_delta_x, mae_delta_y, mae_delta, mae_r

except NameError:
    print('Detectron2 not imported. Predictor class will be unavailable.')

def FourierShift2D(x, delta):
    """
    FourierShift2D(x, delta)
        Subpixel shifting in python. Based on the original script (FourierShift2D.m)
        by Tim Hutt.
        
        Original Description
        --------------------
        Shifts x by delta cyclically. Uses the fourier shift theorem.
        Real inputs should give real outputs.
        By Tim Hutt, 26/03/2009
        Small fix thanks to Brian Krause, 11/02/2010
        
    Parameters
    ----------
    `x`: Numpy Array, required
        The 2D matrix that is to be shifted. Can be real/complex valued.
        
    `delta`: Iterable, required
        The amount of shift to be done in x and y directions. The 0th index should be 
        the shift in the x direction, and the 1st index should be the shift in the y
        direction.
        
        For e.g., For a shift of +2 in x direction and -3 in y direction,
            delta = [2, -3]
        
    Returns
    -------
    `y`: The input matrix `x` shifted by the `delta` amount of shift in the
        corresponding directions.
    """
    # The size of the matrix.
    N, M = x.shape
    
    # Apodisation
#     w_y = signal.tukey(N)[None, :]
#     w_x = signal.tukey(M)[:, None]
    
    # FFT of our possibly padded input signal.
    X = np.fft.fft2(x)
    
    # The mathsy bit. The floors take care of odd-length signals.
    y_arr = np.hstack([
        np.arange(np.floor(N/2), dtype=np.int),
        np.arange(np.floor(-N/2), 0, dtype=np.int)
    ])

    x_arr = np.hstack([
        np.arange(np.floor(M/2), dtype=np.int),
        np.arange(np.floor(-M/2), 0, dtype=np.int)
    ])

    y_shift = np.exp(-1j * 2 * np.pi * delta[0] * x_arr / N)
    x_shift = np.exp(-1j * 2 * np.pi * delta[1] * y_arr / M)

    y_shift = y_shift[None, :] # * w_y  # Shape = (1, N)
    x_shift = x_shift[:, None] # * w_x  # Shape = (M, 1)
#     print(y_arr.shape, y_shift.shape, x_arr.shape, x_shift.shape)
    
    # Force conjugate symmetry. Otherwise this frequency component has no
    # corresponding negative frequency to cancel out its imaginary part.
    if np.mod(N, 2) == 0:
        x_shift[N//2] = np.real(x_shift[N//2])

    if np.mod(M, 2) == 0:
        y_shift[:, M//2] = np.real(y_shift[:, M//2])

    Y = X * (x_shift * y_shift)
    
    # Invert the FFT.
    y = np.fft.ifft2(Y)
    
    # There should be no imaginary component (for real input
    # signals) but due to numerical effects some remnants remain.
    if np.isrealobj(x):
        y = np.real(y)
    
    return y
