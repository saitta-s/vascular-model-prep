import sys
import os
import os.path as osp
import argparse
import pyvista as pv
from time import time



##------------------------ Settings
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_aorta_model_file', type=str, help='Path to the aortic model input file. Can be a PolyData or a Nifti binary label.')
parser.add_argument('--input_lv_model_file', type=str, help='Path to the (optional) left ventricle model input file. Can be a PolyData or a Nifti binary label.')
parser.add_argument('-o', '--output_dir', default='./', type=str, help='Output directory.')
parser.add_argument('--save_name', type=str, help='Output file basename. If None, the input file basename followed by "_prep" will be used.')
parser.add_argument('--show_plots', action='store_true', help='Decide whether to show plots during script execution.')
args = parser.pars_args()

tic = time()


##------------------------ Read input file(s)

