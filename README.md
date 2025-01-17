# vascular-model-prep
Simple code utilities for preparing vascular models for numerical simulations. Developed for the thoracic aorta and coronary arteries, but can easily be applied to other vascular districts.

## Citing
If you use this project in your research, please cite our paper:
```
@article{saitta2022deep,
  title={A deep learning-based and fully automated pipeline for thoracic aorta geometric analysis and planning for endovascular repair from computed tomography},
  author={Saitta, Simone and Sturla, Francesco and Caimi, Alessandro and Riva, Alessandra and Palumbo, Maria Chiara and Nano, Giovanni and Votta, Emiliano and Corte, Alessandro Della and Glauber, Mattia and Chiappino, Dante and others},
  journal={Journal of digital imaging},
  volume={35},
  number={2},
  pages={226--239},
  year={2022},
  publisher={Springer}
}
```

## Features
### Thoracic aorta. 
The code expects two input files corresponding to binary labels of the aorta and the left ventricle. These should be speficied in the main script in the `applications` folder.
The main following operations are then performed. 
- Surface extraction and high quality surface remeshing and smoothing.
- Detection of centerline key points: inlet and outlet points are automatically detected 
using iterative thresholding on the geodesic distances.
- Centerline computation using `vmtk`.
- Inlet opening and extension. Useful for CFD applications.
- Outlet opening and edge tagging.

### Coronary arteries
TODO

## Usage

### Environment preparation

```bash
# Create a conda environment with python 3.10:
conda create -n vmp python=3.10 anaconda

# Install vmtk first:
conda config --add channels conda-forge
conda config --set channel_priority strict
conda install vmtk

# Install requirements
pip install -r requirements.txt

# For interactive plotting, depending on your OS and kernel, you might need to:
pip install 'jupyterlab>=3' ipywidgets 'pyvista[all,trame]'
```