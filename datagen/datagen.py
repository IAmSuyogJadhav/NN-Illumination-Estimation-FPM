import sys
import os
import glob
from tqdm import tqdm
from utils import generate_labels_csv
import FPMSimulation as fpm
sim = fpm.initialize()

args = sys.argv
assert len(args) == 3, "Usage: python datagen.py in_path out_path"
_, in_path, out_path = sys.argv
amps = glob.glob(os.path.join(in_path, '*/amplitude/*.tiff')) + glob.glob(os.path.join(in_path, '*/amplitude/*.tif'))

zernikeCoefficients = 'zernikeCoefficientsOnAxis.txt'
simulationParameters = 'simulationParameters.txt'

os.makedirs(os.path.join(out_path, 'images'), exist_ok=True)
os.makedirs(os.path.join(out_path, 'labels'), exist_ok=True)

for amp in tqdm(amps):
    phase = amp.replace('amplitude', 'phase')
    if not os.path.exists(phase):
        print(f'\r[W] No phase file found corresponding to {amp}. Skipping...')
    else:
        sim.makeSimulatedData(
            amp,
            phase,
            os.path.join(out_path, 'images', os.path.basename(amp)),
            os.path.join(out_path, 'labels', os.path.splitext(os.path.basename(amp))[0] + '.mat'),
            simulationParameters,
            zernikeCoefficients,
            False,
            nargout=0
            )

print('Generating labels.csv (paths relative to current directory)')
generate_labels_csv(os.path.join(out_path, 'images'), os.path.join(out_path, 'labels.csv'))

#files = glob.glob(os.path.join(out_path, 'images', '*.tif')) + glob.glob(os.path.join(out_path, 'images', '*.tiff'))

#d = {
#    'file': []
#}

#dicts = [(f, loadmat(f.replace('tiff', 'mat').replace('tif', 'mat').replace('images', 'labels'))) for f in files]
#ignore_keys = ['__header__', '__version__', '__globals__']

#for f, item in dicts:
#    # Filename
#    d['file'].append(f.replace('mat', 'tif'))
#    for key, value in item.items():
#        if key in ignore_keys:
#            continue
#        # Squeeze numpy arrays
#        if isinstance(value, np.ndarray):
#            value = value.squeeze()
#        
#        # Merge all other keys
#        if key not in d:
#            d[key] = []
#        
#        d[key].append(value)
#
#df = pd.DataFrame(d)
#df.to_csv(os.path.join(out_path, 'labels.csv'), index=False)
