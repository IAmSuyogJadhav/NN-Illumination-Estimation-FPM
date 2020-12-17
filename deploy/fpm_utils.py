import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cv2
from tqdm.auto import tqdm
from utils import read_tiff, FourierShift2D  # from eval_utils import read_tiff, FourierShift2D

plt.rcParams['image.cmap'] = 'Greys_r'
F = lambda x: cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(x)))
Ft = lambda x: cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(x)))


def opts():
    plt.xticks([])
    plt.yticks([])
    

def correct(img, shift=True):
    """Correct the orientation of the obtained image."""
    if shift:
        return cp.fliplr(cp.flipud(cp.fft.fftshift(img)))
    else:
        return cp.fliplr(cp.flipud(img))


def metric_norm(Ip, o_f_i, p):
    """
    Metric from Tian et. al., with normalization.
    """
    if not isinstance(Ip, list):
        Ip = [Ip]
    
    # rec = Reconstructed Image Intensity
    rec = correct(cp.abs(cp.fft.fft2(o_f_i * p)) ** 2)
    loss = 0
    pxs = 0
    for I in Ip:
        loss += ((I - rec) ** 2).sum()
        pxs += I.size
    return loss / pxs

def conv_idx_norm(I_l, I_l_m):
    """
    Convergence Index, with normalization.
    """
    if not isinstance(I_l, list):
        I_l = [I_l]
        
    if not isinstance(I_l_m, list):
        I_l_m = [I_l_m]
    
    assert len(I_l) == len(I_l_m), "Both I_l should be equal in length"
    
    loss = 0
    for il, ilm in zip(I_l, I_l_m):
        loss += cp.sqrt(il).mean() / (abs(cp.sqrt(ilm) - cp.sqrt(il))).sum()
        
    return loss / len(I_l)


def to_uint8(img):
    """Rescale the image to 0-255 and return as uint8"""
    img = img.astype(cp.float32)
    return (255 * (img - img.min()) / (img.max() - img.min())).astype(cp.uint8)


def inv_conv_idx(I_l, I_l_m):
    """
    Modified convergence index, with normalization. Ranges between 0-1. Lower the better.
    """
    if not isinstance(I_l, list):
        I_l = [I_l]
        
    if not isinstance(I_l_m, list):
        I_l_m = [I_l_m]
    
    assert len(I_l) == len(I_l_m), "Both I_l should be equal in length"
    
    # Make sure the images are in the same range of values
    I_l = [to_uint8(im).astype(cp.float32) for im in I_l]
    I_l_m = [to_uint8(im).astype(cp.float32) for im in I_l_m]
    
    loss = 0
    pxs = 0
    for il, ilm in zip(I_l, I_l_m):
        loss += (abs(cp.sqrt(ilm) - cp.sqrt(il))).sum() / cp.sqrt(ilm).mean()
        pxs += il.size
    return loss / (len(I_l) * pxs)


def get_bg(imgs, percent=0.1, tol=100):
    """
    Subtract background noise (adapted from Tian et. al)
    """
    assert 0 < percent < 1, "Percent should be less than 1 and greater than 0."
    
    imsize = imgs[0].shape[0]
    perc = int(percent * imsize)

    # Take 4 corners' mean
    mean = cp.array([
        cp.mean(cp.array([
            img[(1-i) * (imsize-perc):i * perc + (1-i) * imsize, (1-j) * (imsize-perc):j * perc + (1-j) * imsize]
            for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]
        ]))
        for img in imgs
    ])
    # Above code is equivalent to
    # m1 = cp.array([cp.mean(img[:perc, :perc]) for img in imgs])
    # m2 = cp.array([cp.mean(img[:perc, imsize-perc:]) for img in imgs])
    # m3 = cp.array([cp.mean(img[imsize-perc:, :perc]) for img in imgs])
    # m4 = cp.array([cp.mean(img[imsize-perc:, imsize-perc:]) for img in imgs])
    # mean = (m1+m2+m3+m4) / 4 

    # Ignore very high values (usually caused in empty frames)
    mean[mean > tol] = mean[mean<tol].mean()
    return mean


def get_cutoff(row):
    """Calculate cutoff frequency"""
    NA = row.NA
    PIXELSIZE = int(row.PIXELSIZE)
    IMAGESIZE = int(row.IMAGESIZE)
    RI = float(row.RI)
    MAGNIFICATION = int(row.MAGNIFICATION)
#     ILLUMINATION_OFFCENTER_X = float(row.ILLUMINATION_OFFCENTER_X)
#     ILLUMINATION_OFFCENTER_Y = float(row.ILLUMINATION_OFFCENTER_Y)
    WAVELENGTH = float(row.WAVELENGTH)    
#     FRAMES = int(row.FRAMES)

    NYQUIST_FREQ = 2 * np.pi / (2 * PIXELSIZE / (RI * MAGNIFICATION))  # formula
    CUTOFF_FREQ = 2 * np.pi * NA * RI / WAVELENGTH  # formula
    CUTOFF_FREQ_px = (CUTOFF_FREQ / NYQUIST_FREQ) * (IMAGESIZE / 2)
    
    return CUTOFF_FREQ_px


def o_f1(imgs, hres_size, row=None, do_fil=False, show=False):
    """
    Fourier Spectrum Initialization #1
    ----------------------------------
    Mean Image > sqrt > Ft > (optional) filter Ft with 2 * cutoff_freq > pad Ft > Return Ft
    """
    im = cp.array(imgs).mean(0)
    f = Ft(cp.sqrt(im))
    
    if do_fil:
        _orig = int(im.shape[0]) // 2 - 1
        CUTOFF_FREQ_px = get_cutoff(row)
        fil = np.zeros((int(im.shape[0]), int(im.shape[0])))
        fil = cp.array(cv2.circle(fil, (_orig, _orig), 2*CUTOFF_FREQ_px, 1, -1))
        f = f * fil

    pad = (hres_size[0] - imgs[0].shape[0]) // 2
    f = cp.pad(f, [(pad, pad), (pad, pad)])
    
    if show:
        plt.imshow(cp.asnumpy(cp.log(abs(f) + 1e-7)))
        plt.title(f'o_f1 {"without" if not do_fil else "with"} filtering')
        plt.show()
    
    return f

def o_f2(imgs, hres_size, row=None, do_fil=False, show=False):
    """
    Fourier Spectrum Initialization #2
    ----------------------------------
    Mean Image > pad with reflect padding > sqrt > Ft > (optional) filter Ft with 2 * cutoff_freq > return Ft
    """
    im = cp.array(imgs).mean(0)
    pad = (hres_size[0] - imgs[0].shape[0]) // 2
    im = cp.pad(cp.array(im), [(pad, pad), (pad, pad)], mode='reflect')
    f = Ft(cp.sqrt(im))
    
    if do_fil:
        _orig = hres_size[0] // 2 - 1
        CUTOFF_FREQ_px = get_cutoff(row)
        fil = np.zeros(hres_size)
        fil = cp.array(cv2.circle(fil, (_orig, _orig), 2*CUTOFF_FREQ_px, 1, -1))
        f = f * fil
    if show:
        plt.imshow(cp.asnumpy(cp.log(abs(f) + 1e-7)))
        plt.title(f'o_f2 {"without" if not do_fil else "with"} filtering')
        plt.show()
    
    return f

def o_f3(imgs, hres_size, row=None, do_fil=False, show=False):
    """
    Fourier Spectrum Initialization #3
    ----------------------------------
    Mean Image > pad with reflect padding > sqrt > Ft > (optional) filter Ft with 2 * cutoff_freq > Return Ft
    """
    im = cp.array(imgs).mean(0)
    pad = (hres_size[0] - imgs[0].shape[0]) // 2
    im = cp.pad(cp.array(im), [(pad, pad), (pad, pad)], mode='reflect')
    f = Ft(cp.sqrt(im))
    
    if do_fil:
        _orig = hres_size[0] // 2 - 1
        CUTOFF_FREQ_px = get_cutoff(row)
        fil = np.zeros(hres_size)
        fil = cp.array(cv2.circle(fil, (_orig, _orig), 2*CUTOFF_FREQ_px, 1, -1))
        f = f * fil

    if show:
        plt.imshow(cp.asnumpy(cp.log(abs(f) + 1e-7)))
        plt.title(f'o_f3 {"without" if not do_fil else "with"} filtering')
        plt.show()
    
    return f


def reconstruct(imgs, discs, hres_size, row, n_iters=1, o_f_init=None, del_1=1000, del_2=1, round_values=True, plot_per_frame=False, show_interval=None, subtract_bg=False, out_path=None):
    """The main reconstruction algorithm. Adapted from Tian et. al."""
    # Put input images on GPU, estimate background noise
    imgs = [cp.array(img) for img in imgs]
    bgs = get_bg(imgs) if subtract_bg else cp.zeros(len(imgs))
    
    IMAGESIZE = imgs[0].shape[0]
    CUTOFF_FREQ_px = get_cutoff(row)
    FRAMES = len(imgs)
    
    orig = IMAGESIZE // 2 - 1  # Low-res origin
    lres_size = (IMAGESIZE, IMAGESIZE)
    m1, n1 = lres_size
    m, n = hres_size

    losses = []  # Reconstruction Loss
    convs = []  # Inverse Convergence index 

    # Initial high-res guess
    if lres_size==hres_size:  # Initialize with ones
        # Use old algorithm
        F = lambda x: cp.fft.fftshift(cp.fft.fft2(x))
        Ft = lambda x: cp.fft.ifft2(cp.fft.ifftshift(x))
        o = cp.ones(hres_size)
        o_f = F(o)
    elif o_f_init is not None:  # Initialize with given initialization
        F = lambda x: cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(x)))
        Ft = lambda x: cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(x)))
        o = cp.zeros_like(o_f_init)
        o_f = o_f_init
    else:  # Intialize with resized first frame from imgs
        F = lambda x: cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(x)))
        Ft = lambda x: cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(x)))
        o = cp.sqrt(cp.array(cv2.resize(cp.asnumpy(imgs[0] - bgs[0]), hres_size)))
        o_f = Ft(o)

    # Pupil Function
    p = cp.zeros(lres_size)
    p = cp.array(cv2.circle(cp.asnumpy(p), (orig, orig), CUTOFF_FREQ_px, 1, -1))
    ctf = p.copy()  # Ideal Pupil, for filtering later on

    # Main Loop
    log = tqdm(total=n_iters, desc=f'Starting...', bar_format='{percentage:3.0f}% [{elapsed}<{remaining} ({rate_inv_fmt})]{bar}{desc}', leave=False, ascii=True)

    for j in range(n_iters):
        conv = []  # Convergence Index
        for i in range(FRAMES):

            if discs[i] == 0:  # Empty frame
                continue

            # Get k0x, k0y and hence, shifting values
            k0x, k0y = discs[i]

            # Construct auxillary functions for the set of LEDs (= 1, here)
            if hres_size == lres_size:
                shift_x, shift_y = [round(k0x - orig), round(k0y - orig)] if round_values else [k0x - orig, k0y - orig]

                if not round_values:
                    o_f_i = FourierShift2D(o_f, [shift_x, shift_y])  # O_i(k - k_m)
                else:
                    o_f_i = cp.roll(o_f, int(shift_y), axis=0)
                    o_f_i = cp.roll(o_f_i, int(shift_x), axis=1)

                yl, xl = 0, 0   # To reduce code later on

            else:  # Output size larger than individual frames
                _orig = hres_size[0] // 2 - 1
                
                del_x, del_y = k0x - orig, k0y - orig
                x, y = round(_orig + del_x), round(_orig + del_y)

                yl = int(y - m1 // 2)
                xl = int(x - n1 // 2)

                assert xl > 0 and yl > 0, 'Both should be > 0'
                o_f_i = o_f[yl:yl + n1, xl:xl + m1].copy()

            psi_k = o_f_i * p

            # Plot outputs after each frame, for debugging
            if plot_per_frame:
                o_i = Ft(o_f_i * p)
                plt.figure(figsize=(10, 2)); plt.subplot(161); plt.imshow(cp.asnumpy(correct(abs(o_i)))); plt.title(f'$I_{{l}}({i})$'); opts() #DEBUG
                plt.subplot(162); plt.imshow(cp.asnumpy(cv2.convertScaleAbs(cp.asnumpy(20*cp.log(1+abs(o_f_i * p)))))); plt.title(f'$S_{{l}}({i})$'); opts() #DEBUG

            # Impose intensity constraint and update auxillary function
            psi_r = F(psi_k)  #DEBUG: CHANGE BACK TO F
            
            # Low-res estimate obtained from our reconstruction
            I_l = abs(psi_r) if lres_size != hres_size else abs(psi_r)

            # Subtract background noise and clip values to avoid NaN
            I_hat = cp.clip(imgs[i] - bgs[i], a_min=0)
            phi_r = cp.sqrt(I_hat / (cp.abs(psi_r) ** 2)) * psi_r

            phi_k = Ft(phi_r)  #DEBUG: CHANGE BACK TO Ft

            # Update object and pupil estimates
            if hres_size == lres_size:
                if not round_values:
                    p_i = FourierShift2D(p, [-shift_x, -shift_y])  # P_i(k+k_m)      
                else:
                    p_i = cp.roll(p, int(-shift_y), axis=0)
                    p_i = cp.roll(p_i, int(-shift_x), axis=1)

                if not round_values:
                    phi_k_i = FourierShift2D(phi_k, [-shift_x, -shift_y])  # Phi_m_i(k+k_m)
                else:
                    phi_k_i = cp.roll(phi_k, int(-shift_y), axis=0)
                    phi_k_i = cp.roll(phi_k_i, int(-shift_x), axis=1)
            else:  # Output size larger than individual frames
                p_i = p.copy()
                phi_k_i = phi_k.copy()

            ## O_{i+1}(k)
            temp = o_f[yl:yl + n1, xl:xl + m1].copy() + ( cp.abs(p_i) * cp.conj(p_i) * (phi_k_i - o_f[yl:yl + n1, xl:xl + m1].copy() * p_i) ) / \
                        ( cp.abs(p).max() * (cp.abs(p_i) ** 2 + del_1) )

            ## P_{i+1}(k)
            p   =  p  + ( cp.abs(o_f_i) * cp.conj(o_f_i) * (phi_k - o_f_i * p) ) / \
                        ( cp.abs(o_f[yl:yl + n1, xl:xl + m1].copy()).max() * (cp.abs(o_f_i) ** 2 + del_2) )

            o_f[yl:yl + n1, xl:xl + m1] = temp.copy()

            ###### Using F here instead of Ft to get upright image
            o = F(o_f) if lres_size != hres_size else Ft(o_f)
            ######

            if plot_per_frame:
                plt.subplot(163); plt.imshow(cp.asnumpy(cp.mod(ctf*cp.angle(p), 2*cp.pi))); plt.title(f'P({i})'); opts() #DEBUG
                plt.subplot(164); plt.imshow(cp.asnumpy(correct(abs(o)))); plt.title(f'$I_{{h}}({i})$'); opts() #DEBUG
                plt.subplot(165); plt.imshow(cp.asnumpy(correct(cp.angle(o)))); plt.title(f'$\\theta(I_{{h}}({i}))$'); opts() #DEBUG
                plt.subplot(166); plt.imshow(cp.asnumpy(show(cp.asnumpy(o_f)))); plt.title(f'$S_{{h}}({i})$'); opts(); plt.show() #DEBUG

            c = inv_conv_idx(I_l, imgs[i])
            conv.append(c)

        if not plot_per_frame and (show_interval is not None and j % show_interval == 0):
            o_i = Ft(o_f_i * p)  #DEBUG
            plt.figure(figsize=(10, 2)); plt.subplot(161); plt.imshow(cp.asnumpy(correct(abs(o_i)))); plt.title(f'$I_{{l}}({i})$'); opts() #DEBUG
            plt.subplot(162); plt.imshow(cp.asnumpy(cv2.convertScaleAbs(cp.asnumpy(20*cp.log(1+abs(o_f_i * p)))))); plt.title(f'$S_{{l}}({i})$'); opts() #DEBUG
            plt.subplot(163); plt.imshow(cp.asnumpy(cp.mod(ctf*cp.angle(p), 2*cp.pi))); plt.title(f'P({i})'); opts() #DEBUG
            plt.subplot(164); plt.imshow(cp.asnumpy(correct(abs(o)))); plt.title(f'$I_{{h}}({i})$'); opts() #DEBUG
            plt.subplot(165); plt.imshow(cp.asnumpy(correct(cp.angle(o)))); plt.title(f'$\\theta(I_{{h}}({i}))$'); opts() #DEBUG
            plt.subplot(166); plt.imshow(cp.asnumpy(cv2.convertScaleAbs(cp.asnumpy(20*cp.log(1+abs(o_f)))))); plt.title(f'$S_{{h}}({i})$'); opts(); plt.show() #DEBUG

        loss = metric_norm(imgs, o_f_i, p)
        losses.append(loss)
        conv = float(sum(conv) / len(conv))
        convs.append(conv)
        log.set_description_str(f'[Iteration {j + 1}] Convergence Loss: {cp.asnumpy(conv):e}')
        log.update(1)

    scale = 7
    plt.figure(figsize=(3*scale, 4*scale))

    plt.subplot(421)
    plt.plot(cp.asnumpy(cp.arange(len(losses))), cp.asnumpy(cp.clip(cp.array(losses), a_min=None, a_max=1e4)), 'b-')
    plt.title('Loss Curve')
    plt.ylabel('Loss Value')
    plt.xlabel('Iteration')
    plt.subplot(422)
    plt.plot(cp.asnumpy(cp.arange(len(convs))), cp.asnumpy(cp.clip(cp.array(convs), a_min=None, a_max=1e14)), 'b-')
    plt.title('Convergence Index Curve')
    plt.ylabel('Convergence Index')
    plt.xlabel('Iteration')

    amp = cp.array(cv2.resize(read_tiff(row.AMPLITUDE.values[0])[0], hres_size))
    phase = cp.array(cv2.resize(read_tiff(row.PHASE.values[0])[0], hres_size))

    plt.subplot(434)
    plt.title(f'amplitude (Scaled up from {lres_size})')
    plt.imshow(cp.asnumpy(to_uint8(amp))); opts()

    plt.subplot(435)
    plt.title(f'phase (Scaled up from {lres_size})')
    plt.imshow(cp.asnumpy(to_uint8(phase)))

    plt.subplot(436)
    rec = abs(cp.sqrt(amp) * cp.exp(1j * phase))
    plt.title(f'Ground Truth (Scaled up from {lres_size})')
    plt.imshow(cp.asnumpy(to_uint8(rec)))

    plt.subplot(437)
    plt.title('Reconstruction Amplitude')
    amp = abs(o)
    if lres_size == hres_size:
        amp = correct(amp)
    plt.imshow(cp.asnumpy(to_uint8((amp))))

    plt.subplot(438)
    plt.title('Reconstruction Phase')
    phase = cp.angle(o)
    if lres_size == hres_size:
        phase = correct(phase)
    plt.imshow(cp.asnumpy(to_uint8(phase)))

    plt.subplot(439)
    plt.title('Reconstructed Image')
    rec = abs(cp.sqrt(amp) * cp.exp(1j * phase))
    plt.imshow(cp.asnumpy(to_uint8(rec)))

    plt.subplot(427)
    plt.title(f'Recovered Pupil')
    p_show = cp.mod(ctf*cp.angle(p), 2*cp.pi)
    p_show = (p_show / p_show.max() * 255).astype(np.uint8)
    plt.imshow(cp.asnumpy(p_show), cmap='nipy_spectral')

    plt.subplot(428)
    plt.title(f'Raw frames\' mean (Scaled up from {lres_size})')
    plt.imshow(cv2.resize(cp.asnumpy(cp.array(imgs).mean(axis=0)), hres_size))
    
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, bbox_inches='tight')
        plt.close('all')

    # Ignore early noise and print where the error is lowest
    if n_iters > 10:
        it = cp.argmin(cp.array(convs[10:])) + 11
        if out_path is not None:
            print(f'Convergence index lowest at {it}th iteration.')
    else:
        it = cp.argmin(cp.array(convs)) + 1
        if out_path is not None:
            print(f'Convergence index lowest at {it}th iteration.')

    if lres_size == hres_size:
        o = correct(o)
    return o, p, it



def reconstruct_alt(imgs, discs, hres_size, row, n_iters=1, o_f_init=None, del_1=1000, del_2=1, round_values=True, plot_per_frame=False, show_interval=None, subtract_bg=False, out_path=None):
    """The main reconstruction algorithm. Adapted from Tian et. al."""
    # Put input images on GPU, estimate background noise
    imgs = [cp.array(img) for img in imgs]
    bgs = get_bg(imgs) if subtract_bg else cp.zeros(len(imgs))
    
    IMAGESIZE = imgs[0].shape[0]
    CUTOFF_FREQ_px = get_cutoff(row)
    FRAMES = len(imgs)
    
    orig = IMAGESIZE // 2 - 1  # Low-res origin
    lres_size = (IMAGESIZE, IMAGESIZE)
    m1, n1 = lres_size
    m, n = hres_size

    losses = []  # Reconstruction Loss
    convs = []  # Inverse Convergence index 

    # Initial high-res guess
    if lres_size==hres_size:  # Initialize with ones
        # Use old algorithm
        F = lambda x: cp.fft.fftshift(cp.fft.fft2(x))
        Ft = lambda x: cp.fft.ifft2(cp.fft.ifftshift(x))
        o = cp.ones(hres_size)
        o_f = F(o)
    elif o_f_init is not None:  # Initialize with given initialization
        F = lambda x: cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(x)))
        Ft = lambda x: cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(x)))
        o = cp.zeros_like(o_f_init)
        o_f = o_f_init
    else:  # Intialize with resized first frame from imgs
        F = lambda x: cp.fft.fftshift(cp.fft.fft2(cp.fft.ifftshift(x)))
        Ft = lambda x: cp.fft.fftshift(cp.fft.ifft2(cp.fft.ifftshift(x)))
        o = cp.sqrt(cp.array(cv2.resize(cp.asnumpy(imgs[0] - bgs[0]), hres_size)))
        o_f = Ft(o)

    # Pupil Function
    p = cp.zeros(lres_size)
    p = cp.array(cv2.circle(cp.asnumpy(p), (orig, orig), CUTOFF_FREQ_px, 1, -1))
    ctf = p.copy()  # Ideal Pupil, for filtering later on

    # Main Loop
    log = tqdm(total=n_iters, desc=f'Starting...', bar_format='{percentage:3.0f}% [{elapsed}<{remaining} ({rate_inv_fmt})]{bar}{desc}', leave=False, ascii=True)

    for j in range(n_iters):
        conv = []  # Convergence Index
        for i in range(FRAMES):

            if discs[i] == 0:  # Empty frame
                continue

            # Get k0x, k0y and hence, shifting values
            k0x, k0y = discs[i]

            # Construct auxillary functions for the set of LEDs (= 1, here)
            if hres_size == lres_size:
                shift_x, shift_y = [- round(k0x - orig), - round(k0y - orig)] if round_values else [-(k0x - orig), -(k0y - orig)]

                if not round_values:
                    o_f_i = FourierShift2D(o_f, [shift_x, shift_y])  # O_i(k - k_m)
                else:
                    o_f_i = cp.roll(o_f, int(shift_y), axis=0)
                    o_f_i = cp.roll(o_f_i, int(shift_x), axis=1)

                yl, xl = 0, 0   # To reduce code later on

            else:  # Output size larger than individual frames
                _orig = hres_size[0] // 2 - 1
                
                del_x, del_y = k0x - orig, k0y - orig
                x, y = round(_orig - del_x), round(_orig - del_y)

                yl = int(y - m1 // 2)
                xl = int(x - n1 // 2)

                assert xl > 0 and yl > 0, 'Both should be > 0'
                o_f_i = o_f[yl:yl + n1, xl:xl + m1].copy()

            psi_k = o_f_i * p * ctf  #DEBUG: REPLACE * ctf with * p

            # Plot outputs after each frame, for debugging
            if plot_per_frame:
                o_i = Ft(o_f_i * p)
                plt.figure(figsize=(10, 2)); plt.subplot(161); plt.imshow(cp.asnumpy(correct(abs(o_i)))); plt.title(f'$I_{{l}}({i})$'); opts() #DEBUG
                plt.subplot(162); plt.imshow(cp.asnumpy(cv2.convertScaleAbs(cp.asnumpy(20*cp.log(1+abs(o_f_i * p)))))); plt.title(f'$S_{{l}}({i})$'); opts() #DEBUG

            # Impose intensity constraint and update auxillary function
            psi_r = F(psi_k)  #DEBUG: CHANGE BACK TO F
            
            # Low-res estimate obtained from our reconstruction
            I_l = abs(psi_r) if lres_size != hres_size else abs(psi_r)

            # Subtract background noise and clip values to avoid NaN
            I_hat = cp.clip(imgs[i] - bgs[i], a_min=0)
            phi_r = cp.sqrt(I_hat / (cp.abs(psi_r) ** 2)) * psi_r

            phi_k = Ft(phi_r)  #DEBUG: CHANGE BACK TO Ft

            # Update object and pupil estimates
            if hres_size == lres_size:
                if not round_values:
                    p_i = FourierShift2D(p, [-shift_x, -shift_y])  # P_i(k+k_m)      
                else:
                    p_i = cp.roll(p, int(-shift_y), axis=0)
                    p_i = cp.roll(p_i, int(-shift_x), axis=1)

                if not round_values:
                    phi_k_i = FourierShift2D(phi_k, [-shift_x, -shift_y])  # Phi_m_i(k+k_m)
                else:
                    phi_k_i = cp.roll(phi_k, int(-shift_y), axis=0)
                    phi_k_i = cp.roll(phi_k_i, int(-shift_x), axis=1)
            else:  # Output size larger than individual frames
                p_i = p.copy()
                phi_k_i = phi_k.copy()

            ## O_{i+1}(k)
            temp = o_f[yl:yl + n1, xl:xl + m1].copy() + ( cp.abs(p_i) * cp.conj(p_i) * (phi_k_i - o_f[yl:yl + n1, xl:xl + m1].copy() * p_i) ) / \
                        ( cp.abs(p).max() * (cp.abs(p_i) ** 2 + del_1) )

            ## P_{i+1}(k)
            p   =  p  + ( cp.abs(o_f_i) * cp.conj(o_f_i) * (phi_k - o_f_i * p) ) / \
                        ( cp.abs(o_f[yl:yl + n1, xl:xl + m1].copy()).max() * (cp.abs(o_f_i) ** 2 + del_2) )

            o_f[yl:yl + n1, xl:xl + m1] = temp.copy()

            ###### Using F here instead of Ft to get upright image
            o = F(o_f) if lres_size != hres_size else Ft(o_f)
            ######

            if plot_per_frame:
                plt.subplot(163); plt.imshow(cp.asnumpy(cp.mod(ctf*cp.angle(p), 2*cp.pi))); plt.title(f'P({i})'); opts() #DEBUG
                plt.subplot(164); plt.imshow(cp.asnumpy(correct(abs(o)))); plt.title(f'$I_{{h}}({i})$'); opts() #DEBUG
                plt.subplot(165); plt.imshow(cp.asnumpy(correct(cp.angle(o)))); plt.title(f'$\\theta(I_{{h}}({i}))$'); opts() #DEBUG
                plt.subplot(166); plt.imshow(cp.asnumpy(show(cp.asnumpy(o_f)))); plt.title(f'$S_{{h}}({i})$'); opts(); plt.show() #DEBUG

            c = inv_conv_idx(I_l, imgs[i])
            conv.append(c)

        if not plot_per_frame and (show_interval is not None and j % show_interval == 0):
            o_i = Ft(o_f_i * p)  #DEBUG
            plt.figure(figsize=(10, 2)); plt.subplot(161); plt.imshow(cp.asnumpy(correct(abs(o_i)))); plt.title(f'$I_{{l}}({i})$'); opts() #DEBUG
            plt.subplot(162); plt.imshow(cp.asnumpy(cv2.convertScaleAbs(cp.asnumpy(20*cp.log(1+abs(o_f_i * p)))))); plt.title(f'$S_{{l}}({i})$'); opts() #DEBUG
            plt.subplot(163); plt.imshow(cp.asnumpy(cp.mod(ctf*cp.angle(p), 2*cp.pi))); plt.title(f'P({i})'); opts() #DEBUG
            plt.subplot(164); plt.imshow(cp.asnumpy(correct(abs(o)))); plt.title(f'$I_{{h}}({i})$'); opts() #DEBUG
            plt.subplot(165); plt.imshow(cp.asnumpy(correct(cp.angle(o)))); plt.title(f'$\\theta(I_{{h}}({i}))$'); opts() #DEBUG
            plt.subplot(166); plt.imshow(cp.asnumpy(cv2.convertScaleAbs(cp.asnumpy(20*cp.log(1+abs(o_f)))))); plt.title(f'$S_{{h}}({i})$'); opts(); plt.show() #DEBUG

        loss = metric_norm(imgs, o_f_i, p)
        losses.append(loss)
        conv = float(sum(conv) / len(conv))
        convs.append(conv)
        log.set_description_str(f'[Iteration {j + 1}] Convergence Loss: {cp.asnumpy(conv):e}')
        log.update(1)

    scale = 7
    plt.figure(figsize=(3*scale, 4*scale))

    plt.subplot(421)
    plt.plot(cp.asnumpy(cp.arange(len(losses))), cp.asnumpy(cp.clip(cp.array(losses), a_min=None, a_max=1e4)), 'b-')
    plt.title('Loss Curve')
    plt.ylabel('Loss Value')
    plt.xlabel('Iteration')
    plt.subplot(422)
    plt.plot(cp.asnumpy(cp.arange(len(convs))), cp.asnumpy(cp.clip(cp.array(convs), a_min=None, a_max=1e14)), 'b-')
    plt.title('Convergence Index Curve')
    plt.ylabel('Convergence Index')
    plt.xlabel('Iteration')

    amp = cp.array(cv2.resize(read_tiff(row.AMPLITUDE.values[0])[0], hres_size))
    phase = cp.array(cv2.resize(read_tiff(row.PHASE.values[0])[0], hres_size))

    plt.subplot(434)
    plt.title(f'amplitude (Scaled up from {lres_size})')
    plt.imshow(cp.asnumpy(to_uint8(amp))); opts()

    plt.subplot(435)
    plt.title(f'phase (Scaled up from {lres_size})')
    plt.imshow(cp.asnumpy(to_uint8(phase)))

    plt.subplot(436)
    rec = abs(cp.sqrt(amp) * cp.exp(1j * phase))
    plt.title(f'Ground Truth (Scaled up from {lres_size})')
    plt.imshow(cp.asnumpy(to_uint8(rec)))

    plt.subplot(437)
    plt.title('Reconstruction Amplitude')
    amp = abs(o)
    if lres_size == hres_size:
        amp = correct(amp)
    plt.imshow(cp.asnumpy(to_uint8((amp))))

    plt.subplot(438)
    plt.title('Reconstruction Phase')
    phase = cp.angle(o)
    if lres_size == hres_size:
        phase = correct(phase)
    plt.imshow(cp.asnumpy(to_uint8(phase)))

    plt.subplot(439)
    plt.title('Reconstructed Image')
    rec = abs(cp.sqrt(amp) * cp.exp(1j * phase))
    plt.imshow(cp.asnumpy(to_uint8(rec)))

    plt.subplot(427)
    plt.title(f'Recovered Pupil')
    p_show = cp.mod(ctf*cp.angle(p), 2*cp.pi)
    p_show = (p_show / p_show.max() * 255).astype(np.uint8)
    plt.imshow(cp.asnumpy(p_show), cmap='nipy_spectral')

    plt.subplot(428)
    plt.title(f'Raw frames\' mean (Scaled up from {lres_size})')
    plt.imshow(cv2.resize(cp.asnumpy(cp.array(imgs).mean(axis=0)), hres_size))
    
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, bbox_inches='tight')
        plt.close('all')

    # Ignore early noise and print where the error is lowest
    if n_iters > 10:
        it = cp.argmin(cp.array(convs[10:])) + 11
        if out_path is not None:
            print(f'Convergence index lowest at {it}th iteration.')
    else:
        it = cp.argmin(cp.array(convs)) + 1
        if out_path is not None:
            print(f'Convergence index lowest at {it}th iteration.')

    if lres_size == hres_size:
        o = correct(o)
    return o, p, it


####################### RECONSTRUCTION V2 ###############################

cp_inverse_fourier = lambda x: cp.fft.ifftshift(cp.fft.ifft2(cp.fft.fftshift(x)));
cp_forward_fourier = lambda x: cp.fft.ifftshift(cp.fft.fft2(cp.fft.fftshift(x)));

inverse_fourier = lambda x: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)));
forward_fourier = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)));


def o_f4(imgs, hres_size, row=None, do_fil=False, show=False):
    """
    Fourier Spectrum Initialization #1 for V2
    ----------------------------------
    Mean Image > sqrt > Ft > (optional) filter Ft with 2 * cutoff_freq > pad Ft > Return Ft
    """
    im = cp.array(imgs).mean(0)
    f = cp_forward_fourier(cp.sqrt(im))
    
    if do_fil:
        _orig = int(im.shape[0]) // 2 - 1
        CUTOFF_FREQ_px = get_cutoff(row)
        fil = np.zeros((int(im.shape[0]), int(im.shape[0])))
        fil = cp.array(cv2.circle(fil, (_orig, _orig), 2*CUTOFF_FREQ_px, 1, -1))
        f = f * fil

    pad = int((hres_size[0] - imgs[0].shape[0]) // 2)
    f = cp.pad(f, [(pad, pad), (pad, pad)])
    
    if show:
        plt.imshow(cp.asnumpy(cp.log(abs(f) + 1e-7)))
        plt.title(f'o_f1 {"without" if not do_fil else "with"} filtering')
        plt.show()
    
    return f


def show_f(x, title='Figure'):
    """
    Shows Fourier Spectrum as well as the inverse Fourier transform. For debugging.
    """
    plt.suptitle(title)
    plt.subplot(121)
    plt.imshow(cp.asnumpy(cp.log(cp.abs(x) + 1)))

    plt.subplot(122)
    plt.imshow(cp.asnumpy(cp.abs(cp_inverse_fourier(x))))

    plt.show()


def spectral_correlation_calibration_GPU(object_freq_space, experimental_amp,\
                                     recon_pupil, freq_pos, dim_segment_interp, dim_segment):
    """
    [Slightly modified for single-LED case.]
    
    Based on Regina Eckert et.al paper.
    This function will give a correction to the darkfield illumination LEDs
    based on spectral correlatoins. We take the updated object function after
    the m'th iteration and correlate with the spectrum of our low-resolution
    experimental data corresponding to the i'th iteration. This correlation
    will give an estimate of the i'th illumination angle relative to other 
    illumination angles.
    
    Input:
        object_freq_space - the update object function at m+1 iteration
        recon_pupil - updated pupil function at m+1 iteration
        experimental_amp - experimental image amplitude used for comparison
    """
    # our perturbation of k (the amount by which k is shifted) can be defined
    # analytically but I will use 1 pixel for now
    k_shift = 1
#     x1 = int((dim_segment_interp-dim_segment)/2 + freq_pos[1])
#     x2 = int((dim_segment_interp+dim_segment)/2 + freq_pos[1])
#     y1 = int((dim_segment_interp-dim_segment)/2 + freq_pos[0])
#     y2 = int((dim_segment_interp+dim_segment)/2 + freq_pos[0])
    orig = dim_segment // 2 - 1
    _orig = dim_segment_interp // 2 - 1
    k0x, k0y = freq_pos  # Wavevectors
    del_x, del_y = k0x - orig, k0y - orig  # Distance from origin in low-res
    x, y = round(_orig + del_x), round(_orig + del_y)  # Distance from origin in high-res

    xl = int(y - dim_segment // 2)
    yl = int(x - dim_segment // 2)
    x1, x2 =  xl, xl + dim_segment
    y1, y2 =  yl, yl + dim_segment
    
    # store the cost function values
    cost_fn = np.zeros((3,3))
    
    # measured intensity normalized by DC
    experimental_intensity = experimental_amp**2
#    experimental_intensity = experimental_intensity / np.max(experimental_intensity)
    experimental_intensity = experimental_intensity /\
    np.max([cp.mean(experimental_intensity), 1e-10]) - 1
    # we search over a 3x3 space grid at each iteration
    for x in [-1, 0, 1]:
        for y in [-1, 0, 1]:
            
            # shifted low resolution estimate normalized by DC
            shifted_low_res_intensity = cp.abs(cp_inverse_fourier(object_freq_space[x1+ x*k_shift:x2+ x*k_shift, y1 + y*k_shift:y2 + y*k_shift]\
                                                            * recon_pupil))**2
#            shifted_low_res_intensity = shifted_low_res_intensity / np.max(shifted_low_res_intensity)
            shifted_low_res_intensity = shifted_low_res_intensity /\
            np.max([cp.mean(shifted_low_res_intensity), 1e-10]) - 1

            cost_fn[x+1,y+1] = cp.sum(cp.abs(experimental_intensity - shifted_low_res_intensity))
    
    # find the minimum shifts        
    [xmin, ymin] = np.unravel_index(cost_fn.argmin(), cost_fn.shape)
    
    # update the current frequency position array
    freq_pos[1] = freq_pos[1] + (xmin-1)*k_shift
    freq_pos[0] = freq_pos[0] + (ymin-1)*k_shift
    
    return freq_pos
    
    
def reconstruct_v2(
    imgs,
    discs,
    row,
    hres_size,
    n_iters=1,
    do_fil=False,
    denoise=False,
    crop_our_way=True,
    plot=True,
    adaptive_noise=1,
    adaptive_pupil=1,
    adaptive_img=1,
    alpha=1,
#     delta_img=0.1,
#     delta_pupil=1e-6,
    delta_img=10,
    delta_pupil=1e-4,
    eps=1e-9,
    calibrate_freq_pos=False,
    out_path=None
):
    """
    Adapted From Aidukas et al. (2018)
    """    
    # Define basic parameters
    dim_segment_interp = hres_size[0]
    dim_segment = imgs[0].shape[0]
    orig = dim_segment // 2 - 1
    lres_size = (dim_segment, dim_segment)
    clahe_p = cv2.createCLAHE(clipLimit=3, tileGridSize=(10,10))  # For contrast Enhancment
    clahe_a = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(10,10))  # For contrast Enhancment

    # Initialize high-res estimate (freq. domain)
    high_res_freq_estimate = o_f4(imgs, hres_size, row=row, do_fil=do_fil)

    # Pupil and aperture
    CUTOFF_FREQ_px = get_cutoff(row)
    pupil= cp.zeros((dim_segment, dim_segment))
    pupil = cp.array(cv2.circle(cp.asnumpy(pupil), (orig, orig), int(CUTOFF_FREQ_px), 1, -1))
    aperture = pupil.copy() 

    # Progress bar
    log = tqdm(
        total=n_iters,
        desc=f'Working...',
        bar_format='{percentage:3.0f}% [{elapsed}<{remaining} ({rate_inv_fmt})]'
                   '{bar}{desc}',
        leave=False,
        )

    convs = []

    # Main Loop
    for iteration_number in range(n_iters):
        conv = []
        for i in range(len(imgs)):  # Iterate over all the images
            freq_pos = discs[i]
            if freq_pos == 0:  # Skip empty frames
                    continue

            if not crop_our_way:  # Crop with formula used by original author
                x1 = int((dim_segment_interp-dim_segment)/2 + freq_pos[1])
                x2 = int((dim_segment_interp+dim_segment)/2 + freq_pos[1])
                y1 = int((dim_segment_interp-dim_segment)/2 + freq_pos[0])
                y2 = int((dim_segment_interp+dim_segment)/2 + freq_pos[0])
                low_res_freq_estimate = cp.copy(high_res_freq_estimate[x1:x2, y1:y2])
            
            else:  # Use our own (Earlier doesn't seem to work)
                
                # High-res origin
                _orig = hres_size[0] // 2 - 1
                k0x, k0y = freq_pos  # Wavevectors
                del_x, del_y = k0x - orig, k0y - orig  # Distance from origin in low-res
                x, y = round(_orig + del_x), round(_orig + del_y)  # Distance from origin in high-res

                xl = int(y - dim_segment // 2)
                yl = int(x - dim_segment // 2)
                x1, x2 =  xl, xl + dim_segment
                y1, y2 =  yl, yl + dim_segment
                
                # Low-res estimate (Fourier domain)
                low_res_freq_estimate = cp.copy(high_res_freq_estimate[x1:x2, y1:y2])
    #         show_f(low_res_freq_estimate, title='low_res_freq_estimate')

            # Filter with pupil and aperture
            low_res_freq_estimate_filtered = pupil * aperture * low_res_freq_estimate
    #         show_f(low_res_freq_estimate_filtered, title='low_res_freq_estimate_filtered')
            
            # Back to real domain
            low_res_estimate_filtered = cp_inverse_fourier(low_res_freq_estimate_filtered)

            I_l = abs(low_res_estimate_filtered)
            c = inv_conv_idx(I_l, imgs[i])
            conv.append(c)

            # Measured intensity values
            experimental_amp = cp.copy(imgs[i])

            # Skipping sparse sampling (setting bayer = 1)
            bayer = 1
            if denoise:  # Denoise measurements (not required for our case)
                noise = cp.abs(cp.mean(experimental_amp) - cp.mean(cp.abs(low_res_estimate_filtered)**2 * bayer)) * adaptive_noise
                denoised_image = (cp.abs(experimental_amp) - noise) * ((cp.abs(experimental_amp) - noise) > 0)
                denoised_image = cp.sqrt(denoised_image) * (bayer) + cp.abs(low_res_estimate_filtered) * cp.abs(1 - bayer)
            else:
                denoised_image = cp.sqrt(cp.abs(experimental_amp))

            # Update low-res estimate with measured intensity values
            low_res_updated = cp.abs(denoised_image)  \
                                    * low_res_estimate_filtered / (cp.abs(low_res_estimate_filtered)+ eps)
            low_res_freq_updated = cp_forward_fourier(low_res_updated) 
    #         show_f(low_res_freq_updated, 'low_res_freq_updated')

            # Update high-res estimate (Fourier domain)
            temp = aperture * (low_res_freq_updated - low_res_freq_estimate_filtered)

            Omax = cp.max(cp.abs(high_res_freq_estimate))
            high_res_freq_estimate[x1:x2, y1:y2] = low_res_freq_estimate + alpha*adaptive_pupil**(iteration_number+1) \
                                    * cp.abs(pupil) * cp.conj(pupil) * temp \
                                    / (cp.max(cp.abs(pupil)) * (cp.abs(pupil)**2 + delta_img))
    #         show_f(high_res_freq_estimate, 'high_res_freq_estimate')

            # Update Pupil
            pupil = pupil + alpha*adaptive_pupil**(iteration_number+1) \
                            * aperture * cp.abs(low_res_freq_estimate) * cp.conj(low_res_freq_estimate) * temp \
                            / (Omax * (cp.abs(low_res_freq_estimate)**2 + delta_pupil))

            # Spectral Correlation Calibration
            if calibrate_freq_pos:
#                 print(discs[i]) #DEBUG
                discs[i] = spectral_correlation_calibration_GPU(high_res_freq_estimate, denoised_image,\
                                         pupil, freq_pos, dim_segment_interp, dim_segment)
#                 print(discs[i]) #DEBUG
            
            
        # Out of inner loop
        if plot:
            conv = float(sum(conv) / len(conv))
            convs.append(conv)
        
        # Update progeress bar
        log.update()

    # Get high-res estimate
    high_res_estimate = inverse_fourier(high_res_freq_estimate);
    
    
    if plot:
        log.set_description_str('Drawing figure...')
        # Create Figure
        scale = 7
        plt.figure(figsize=(2*scale, 4*scale))

        amp = cp.array(cv2.resize(read_tiff(row.AMPLITUDE.values[0])[0], hres_size, interpolation=cv2.INTER_NEAREST))
        phase = cp.array(cv2.resize(read_tiff(row.PHASE.values[0])[0], hres_size, interpolation=cv2.INTER_NEAREST))

        plt.subplot(421)
        plt.title(f'Intensity (Scaled up from {lres_size})')
        plt.imshow(cp.asnumpy(to_uint8(amp)), interpolation='none'); plt.colorbar()

        plt.subplot(422)
        plt.title(f'Phase (Scaled up from {lres_size})')
        plt.imshow(cp.asnumpy(to_uint8(phase)), interpolation='none'); plt.colorbar()

        plt.subplot(423)
        plt.title('Reconstruction Intensity')
        amp = cp.abs(cp.rot90(high_res_estimate, 0))
        plt.imshow(cp.asnumpy(to_uint8((amp))), interpolation='none'); plt.colorbar()

        plt.subplot(424)
        plt.title('Reconstruction Phase')
        phase = cp.angle(cp.rot90(high_res_estimate, 0))
        plt.imshow(cp.asnumpy(to_uint8(phase)), interpolation='none'); plt.colorbar()

        plt.subplot(425)
#         plt.title('Reconstruction Intensity (Contrast Corrected)')
#         amp = cp.abs(cp.rot90(high_res_estimate, 0))
# #         plt.imshow(cv2.equalizeHist(cp.asnumpy(to_uint8(phase))))
#         plt.imshow(clahe_a.apply(cp.asnumpy(to_uint8(amp))))
        plt.title(f'Mean of Measured Intensity (Scaled up from {lres_size})')
        plt.imshow(cv2.resize(cp.asnumpy(to_uint8(cp.array(imgs).mean(axis=0))), hres_size, interpolation=cv2.INTER_NEAREST), interpolation='none'); plt.colorbar()
        
        
        plt.subplot(426)
        plt.title('Reconstruction Phase (Contrast Corrected)')
        phase = cp.angle(cp.rot90(high_res_estimate, 0))
#         plt.imshow(cv2.equalizeHist(cp.asnumpy(to_uint8(phase))))
        plt.imshow(clahe_p.apply(cp.asnumpy(to_uint8(phase))), interpolation='none'); plt.colorbar()
        
        plt.subplot(427)
        plt.title(f'Recovered Pupil Intensity')
        p_show = cp.asnumpy(cp.abs(pupil))
        plt.imshow(p_show, interpolation='none'); plt.colorbar()

        plt.subplot(428)
        plt.title(f'Recovered Pupil Phase')
        p_show = cp.asnumpy(cp.angle(pupil))
        plt.imshow(p_show, interpolation='none'); plt.colorbar()

#         plt.subplot(4,3,12)
#         plt.title(f'Raw frames\' mean (Scaled up from {lres_size})')
#         plt.imshow(cv2.resize(cp.asnumpy(cp.array(imgs).mean(axis=0)), hres_size, interpolation=cv2.INTER_NEAREST))
        
        if (out_path is None) and plot:
            plt.show()
        else:  # Save figure to the specified path
            plt.savefig(out_path, bbox_inches='tight')
            plt.close('all')
        
    log.set_description_str('Done.')
    log.close()
    return high_res_estimate, pupil