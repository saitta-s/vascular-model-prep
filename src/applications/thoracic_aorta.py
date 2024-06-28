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
from vascular_prep import utils as ut


##------------------------ Settings
"""
parser = argparse.ArgumentParser()
parser.add_argument('--input_aorta_model_file', type=str, help='Path to the aortic model input file. Can be a PolyData or a Nifti binary label.')
parser.add_argument('--input_lv_model_file', type=str, help='Path to the left ventricle model input file. Can be a PolyData or a Nifti binary label.')
parser.add_argument('-o', '--output_dir', default='./', type=str, help='Output directory.')
parser.add_argument('--delta', type=float, default=0.05, help='Delta value for recursive thresholding.')
parser.add_argument('--save_name', type=str, default='output', help='Output file basename. If None, the input file basename followed by "_prep" will be used.')
parser.add_argument('--show_plots', action='store_true', help='Decide whether to show plots during script execution.')
args = parser.parse_args()
"""
tic = time()

input_aorta_model_file = r"/mnt/c/DATA/Work/dev/vascular-model-prep/data/raw/labels/aorta/0001.nii.gz"
input_lv_model_file = r"/mnt/c/DATA/Work/dev/vascular-model-prep/data/raw/labels/left_ventricle/0001.nii.gz"
output_dir = './'
delta = 0.05

##------------------------ Read input file(s)
ao_surf = io.read_file(input_aorta_model_file)

if input_lv_model_file is not None:
    lv_surf = io.read_file(input_lv_model_file)
else:
    raise NotImplementedError()
    #lv_surf = None


##------------------------ Remeshing and smoothing
ao_surf = ut.meshfix(ao_surf)
ao_surf = ut.polydata_smoothing(ao_surf, lamb=0.95, iterations=100)#.smooth(500)
ao_surf = ut.mmg_remesh(ao_surf, hausd=0.5, hmax=1.5, hmin=0.9, max_aspect_ratio=5, max_iter=3, verbose=True)

lv_surf.decimate(0.6)


##------------------------ Detect aortic valve plane (inlet)
print('Source point detection...')
lv_surf.compute_implicit_distance(ao_surf, inplace=True)
av_region = lv_surf.threshold_percent(percent=0.1, scalars='implicit_distance', invert=True)
av_center = np.array(av_region.center)
av_radius = np.mean([np.linalg.norm(pt - av_center) for pt in av_region.points])
av_normal = pv.fit_plane_to_points(av_region.points).point_normals.mean(0)
if av_normal[2] < 0: av_normal = -1 * av_normal


##------------------------ Detect centerline keypoints
print('Centerline seed detection...')
source_pt, _ = ao_surf.ray_trace(av_center-20*av_normal, av_center+20*av_normal)
st_id = ao_surf.find_closest_point(source_pt.T)
source_pt = ao_surf.points[st_id]

# Detect aortic centerline endpoints
G_ao = ut.convert_triangle_mesh_to_graph(ao_surf)
shortest_paths_lengths = nx.single_source_shortest_path_length(G_ao, st_id)
ao_surf['Geodesics'] = np.zeros((ao_surf.n_points,))
for j in range(ao_surf.n_points):
    geo_path = shortest_paths_lengths[j]
    ao_surf['Geodesics'][j] = geo_path
ao_surf['Geodesics_n'] = (ao_surf['Geodesics'] - np.min(ao_surf['Geodesics'])) / np.ptp(ao_surf['Geodesics'])
desc_ao_pt = ao_surf.points[np.argmax(ao_surf['Geodesics_n'])]
surf_with_geodesics = ao_surf.copy()

# Detect ends: recursive thresholding of geodesic distance to ostia
delta = delta
cand_outlets = []
for delta in np.arange(delta, 1, delta):
    thr = ao_surf.threshold_percent(1-delta, scalars='Geodesics_n').connectivity()
    for j in np.unique(thr['RegionId']):
        thr_connected = thr.threshold([j, j+0.5], scalars='RegionId')
        cand_outlets.append(np.array(thr_connected.points[np.argmax(thr_connected['Geodesics_n'])]))
cand_outlets = np.unique(cand_outlets, axis=0)

# descending aorta point
toremove = np.argmin([np.linalg.norm(desc_ao_pt - c) for c in cand_outlets])
cand_outlets = np.array([cand_outlets[i] for i in range(len(cand_outlets)) if i != toremove])


##------------------------ Centerlines extraction
print('Computing centerlines...')
parentCenterline = ut.extract_centerline(ao_surf, list(source_pt), list(desc_ao_pt), resampling=0.1, appendEndPoints=0)

# Branches centerline extraction
centerlines = ut.extract_centerline(ao_surf, list(source_pt), [c for pt in cand_outlets for c in pt],
                                        resampling=0.1, appendEndPoints=0).connectivity()


##------------------------ Proximal clip (with optional extrusion)
print('\nClipping aorta proximally...')
n_cells = []
dists_to_av = []
offset_array = np.arange(-20, 20, 0.5)
slices = []
for i in offset_array:
    disc_ = pv.Disc(center=av_center+i*av_normal, inner=0, outer=av_radius*3, normal=av_normal, r_res=20, c_res=20)
    col, n_contacts = ao_surf.collision(disc_)
    n_cells.append(n_contacts)

selected_center = av_center + offset_array[np.min(np.where(np.array(n_cells) > 0)) - 1] * av_normal

offset = 40
cut_cyl = pv.Cylinder(center=selected_center,
                      radius=av_radius,
                      direction=av_normal,
                      height=offset,
                      resolution=200,
                      capping=True).triangulate().subdivide(2)
cut_cyl_inlet = cut_cyl.copy(deep=True)
clipped_ao = ao_surf.clip_surface(cut_cyl, invert=False)
clipped_ao_clean = pv.PolyData(clipped_ao.points, faces=clipped_ao.faces)

if show_plot:
    pl = pv.Plotter()
    pl.add_mesh(clipped_ao_clean, color='w', opacity=0.8)
    pl.add_points(selected_center, color='m', point_size=15)
    pl.add_mesh(cut_cyl, color='b', opacity=0.6)
    pl.show_axes()
    pl.show()

edges = clipped_ao_clean.extract_feature_edges(80).connectivity()
edge_dists_to_av = []
for i in np.unique(edges['RegionId']):
    thr = edges.threshold([i, i+0.5], scalars='RegionId')
    if thr.n_points > 10:
        edge_dists_to_av.append(np.linalg.norm(thr.center - np.array(av_center)))
    else:
        edge_dists_to_av.append(1000)
idx = np.unique(edges['RegionId'])[np.argmin(edge_dists_to_av)]
edge_2keep = edges.threshold([idx, idx+0.5], scalars='RegionId').extract_surface()

# extrude edge
extr = edge_2keep.extrude(vector=-av_normal*20, capping=False).triangulate()#.subdivide(2)
combined_surf = clipped_ao.merge(extr).clean()

# STITCHING ------------------------------------------------------
#combined_surf_remeshed = pv.wrap(mut.acvdq_remesh_surface(combined_surf, nOfNodes=120000, adaptive=0, manifold=0, keep_boundary=1)).connectivity()
#combined_surf_remeshed = combined_surf_remeshed.smooth(800)
#combined_surf_remeshed.save('ao.vtp')
combined_surf.save(f'ao_{pid}.stl')

mfix = PyTMesh(False)  # False removes extra verbose output
mfix.load_file(f'ao_{pid}.stl')
os.remove(f'ao_{pid}.stl')
mfix.fill_small_boundaries(nbe=100, refine=True)
vert, faces = mfix.return_arrays()
triangles = np.empty((faces.shape[0], 4), dtype=faces.dtype)
triangles[:, -3:] = faces
triangles[:, 0] = 3

mesh = pv.PolyData(vert, triangles)

mesh = pv.wrap(mut.acvdq_remesh_surface(mesh, nOfNodes=120000, adaptive=0, manifold=0, keep_boundary=1, fn=f'{pid}.ply')).connectivity()
#mesh.connectivity().save('ao.vtp')


# clip combined surf
cut_cyl2 = pv.Cylinder(center=selected_center-av_normal*offset/2,
                      radius=av_radius*2,
                      direction=av_normal,
                      height=offset*0.95,
                      resolution=200,
                      capping=True).triangulate().subdivide(4)
cut_cyl_inlet = cut_cyl2.copy(deep=True)
combined_surf = mesh.clip_surface(cut_cyl2, invert=False).connectivity()
combined_surf = pv.wrap(mut.acvdq_remesh_surface(combined_surf, nOfNodes=150000, adaptive=1, manifold=0, keep_boundary=1, fn=f'{pid}.ply')).connectivity()
#combined_surf.save('ao.vtp')




io.save_to_file(ao_surf, osp.join(output_dir, 'aorta.vtp'), clear_data=False)
io.save_to_file(parentCenterline, osp.join(output_dir, 'parentCenterline.vtp'), clear_data=False)
io.save_to_file(centerlines, osp.join(output_dir, 'centerlines.vtp'), clear_data=False)

