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


## Webapp Usage
1. If you have already cloned this repository using a git client or GitHub Desktop in previous steps, simply switch to the ```deploy``` branch and continue to the next step.

```bash
git checkout deploy
```

​	Otherwise, use the drop-down menu towards the top of this page to switch branches from ```master``` to ```deploy``` and download the zip file.

​	In addition to the requirements stated earlier, you will need flask to run the app. Install flask:

```bash
pip install flask
```



2. You will need to have the trained models in order to use the app. The model files will go inside `app/static/models`.
3. Start a bash/CMD/Powershell session inside the directory and run:

```bash
flask run
```

​	Open a new browser window and navigate to http://localhost:5000. Follow the On-screen instructions to navigate the interface.