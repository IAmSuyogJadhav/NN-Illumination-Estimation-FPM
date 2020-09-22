# NN-Illumination-Estimation-FPM
Accompanying code for the paper Neural Network for Illumination Angle Estimation in Fourier Ptychography


## Installation
Install PyTorch, Torchvision and CUDAToolkit (10.2 needed for compatibility with Detectron2, newer versions might give compatibility issues). The following command is for Anaconda package manager, it helps in managing package versions and has a good repository system of its own as well. Anaconda can be downloaded [here](https://www.anaconda.com/products/individual). If you wish to skip this step, you will have to install CUDAToolkit-10.2 separately and then install pytorch + torchvision using the pip command available on their [website](https://pytorch.org/).

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

### Linux Users
The next step for Linux users is to simply install the necessary requirements.

```bash
pip install -r requirements.txt
```

### Windows Users
Detectron2 does not yet officially support Windows. The community has managed to get Detectron2 running on Windows by modifying some of the code.

1. Install Detectron2 by following the instructions given on this well-written [blog](https://medium.com/@dgmaxime/how-to-easily-install-detectron2-on-windows-10-39186139101c).

2. Install rest of the requirements

```bash
pip install -r requirements-win.txt
```


## Usage
TBD