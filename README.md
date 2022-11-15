# NN-Illumination-Estimation-FPM
Accompanying code for the paper "Object Detection Neural Network Improves Fourier Ptychographic Reconstruction" ([Paper](https://doi.org/10.1364/OE.409679)).


## Installation

### Create a New Environment (optional)
To avoid messing up your exisiting setup, it is recommended to create a new virtual environment before you proceed. The following command is for Anaconda package manager, it is a useful package manager with a good repository system of its own that reduces some hassles. Anaconda can be downloaded from [here](https://www.anaconda.com/products/individual). If you don't wish to use Anaconda, you can use [venv](https://docs.python.org/3/library/venv.html#creating-virtual-environments) to achieve the same effect.

```bash 
conda create -n myenv python=3.7
conda activate myenv
```

### Install PyTorch
Install PyTorch, Torchvision and CUDAToolkit (10.2 needed for compatibility with Detectron2, newer versions might give compatibility issues). If you don't wish to use Anaconda, you will have to install CUDAToolkit-10.2 separately and then install pytorch + torchvision using the pip command available on their [website](https://pytorch.org/).

```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

### Detectron2 and Other Requirements

#### For Linux Users
The next step for Linux users is to simply install the necessary requirements.

```bash
pip install -r requirements.txt
```

#### For Windows Users
Detectron2 does not yet officially support Windows. The community has managed to get Detectron2 running on Windows by modifying some of the code.

1. Install Detectron2 by following the instructions given on this well-written [blog](https://medium.com/@dgmaxime/how-to-easily-install-detectron2-on-windows-10-39186139101c).

2. Install rest of the requirements

```bash
pip install -r requirements-win.txt
```


## Usage
1. Clone the GitHub repository using git or download the [zip file](https://github.com/IAmSuyogJadhav/NN-Illumination-Estimation-FPM/archive/refs/heads/master.zip) directly and extract it.

2. Download the pre-trained models from [Dataverse](https://doi.org/10.18710/BBU6JD). Download and extract them in the root directory of the repository. Now you should have a folder named `models` in the root directory containing the pre-trained models.

3. A sample inference code is provided in [`Example.ipynb`](Example.ipynb). You can run it using Jupyter Notebook or Google Colab. The notebook is well commented and should be easy to follow. You can also change the parameters in the notebook to see how they affect the results.

4. A sample input image is provided in the [`tiffs`](tiffs) folder. You can use it to test the code. You can also use your own test images, as long as they are in the multipage tiff format similar to the sample image.
