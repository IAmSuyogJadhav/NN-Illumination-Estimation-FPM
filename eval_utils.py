import pickle
from fvcore.common.file_io import PathManager
from fvcore.common.checkpoint import Checkpointer
import comm
import c2_model_loading
from fvcore.common.registry import Registry
import transform as T
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# META_ARCH_REGISTRY = pickle.load(open('meta_registry.pkl', 'rb'))
META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.
The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model


class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
    model zoo, and apply conversions for legacy models.
    """

    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(
            model,
            save_dir,
            save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
            **checkpointables,
        )

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            model_state_dict = self.model.state_dict()
            c2_model_loading.align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            checkpoint["model"] = model_state_dict
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)
        if incompatible is None:  # support older versions of fvcore
            return None

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        return incompatible

    
class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    If you'd like to do anything more fancy, please refer to its source code
    as examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
#         if len(cfg.DATASETS.TEST):
#             self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        
        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
    
    
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions

        

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