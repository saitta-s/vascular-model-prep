import sys
sys.path.append('../')
sys.path.append('./src')
import os
import os.path as osp
import argparse
import numpy as np
import pyvista as pv
from time import time
import networkx as nx

from vascular_prep import io
from vascular_prep import mesh_utils as mut


##------------------------ Settings
parser = argparse.ArgumentParser()
parser.add_argument('--input_aorta_model_file', type=str, help='Path to the aortic model input file. Can be a PolyData or a Nifti binary label.')
parser.add_argument('--input_lv_model_file', type=str, help='Path to the left ventricle model input file. Can be a PolyData or a Nifti binary label.')
parser.add_argument('-o', '--output_dir', default='./', type=str, help='Output directory.')
parser.add_argument('--delta', type=float, default=0.05, help='Delta value for recursive thresholding.')
parser.add_argument('--save_name', type=str, default='output', help='Output file basename. If None, the input file basename followed by "_prep" will be used.')
parser.add_argument('--show_plots', action='store_true', help='Decide whether to show plots during script execution.')
args = parser.parse_args()

tic = time()


##------------------------ Read input file(s)
ao_surf = io.read_file(args.input_aorta_model_file)

if args.input_lv_model_file is not None:
    lv_surf = io.read_file(args.input_lv_model_file)
else:
    raise NotImplementedError()
    #lv_surf = None


##------------------------ Remeshing and smoothing
ao_surf = mut.meshfix(ao_surf)
ao_surf = ao_surf.decimate(0.8)
ao_surf = mut.polydata_smoothing(ao_surf, lamb=0.8, iterations=500).smooth(500)
ao_surf = mut.pyacvd_remeshing(ao_surf, subdivide=3, cluster=40000)

lv_surf.decimate(0.9)


##------------------------ Detect aortic valve plane (inlet)
print('Source point detection...', end='', flush=True)
lv_surf.compute_implicit_distance(ao_surf, inplace=True)
av_region = lv_surf.threshold_percent(percent=0.1, scalars='implicit_distance', invert=True)
av_center = np.array(av_region.center)
av_radius = np.mean([np.linalg.norm(pt - av_center) for pt in av_region.points])
av_normal = pv.fit_plane_to_points(av_region.points).point_normals.mean(0)
if av_normal[2] < 0: av_normal = -1 * av_normal


##------------------------ Detect centerline keypoints
print('Centerline seed detection...', end='', flush=True)
source_pt, _ = ao_surf.ray_trace(av_center-20*av_normal, av_center+20*av_normal)
st_id = ao_surf.find_closest_point(source_pt.T)
source_pt = ao_surf.points[st_id]

# Detect aortic centerline endpoints
G_ao = mut.convert_triangle_mesh_to_graph(ao_surf)
shortest_paths_lengths = nx.single_source_shortest_path_length(G_ao, st_id)
ao_surf['Geodesics'] = np.zeros((ao_surf.n_points,))
for j in range(ao_surf.n_points):
    geo_path = shortest_paths_lengths[j]
    ao_surf['Geodesics'][j] = geo_path
ao_surf['Geodesics_n'] = (ao_surf['Geodesics'] - np.min(ao_surf['Geodesics'])) / np.ptp(ao_surf['Geodesics'])
desc_ao_pt = ao_surf.points[np.argmax(ao_surf['Geodesics_n'])]
surf_with_geodesics = ao_surf.copy()

# Detect ends: recursive thresholding of geodesic distance to ostia
delta = args.delta
cand_outlets = []
for delta in np.arange(delta, 1, delta):
    thr = ao_surf.threshold_percent(1-delta, scalars='Geodesics_n').connectivity()
    for j in np.unique(thr['RegionId']):
        thr_connected = thr.threshold([j, j+0.5], scalars='RegionId')
        cand_outlets.append(np.array(thr_connected.points[np.argmax(thr_connected['Geodesics_n'])]))
cand_outlets = np.unique(cand_outlets, axis=0)

# descending aorta point
desc_ao_pt = ao_surf.points[np.argmin(ao_surf.points[:, 2])]
toremove = np.argmin([np.linalg.norm(desc_ao_pt - c) for c in cand_outlets])
cand_outlets = np.array([cand_outlets[i] for i in range(len(cand_outlets)) if i != toremove])


io.save_to_file(ao_surf, osp.join(args.output_dir, 'aorta.vtp'), clear_data=True)
io.save_to_file(pv.PolyData(cand_outlets), osp.join(args.output_dir, 'pts.vtp'), clear_data=True)

