import numpy as np
import pyvista as pv
import pymeshfix as mf
import pyacvd
from time import time
import trimesh
import networkx as nx


def meshfix(mesh):
    tic = time()
    print('Meshfix... ', end='', flush=True)
    meshfix = mf.MeshFix(pv.PolyData(mesh))
    meshfix.repair(verbose=False)
    mesh = meshfix.mesh
    print('  {:.2f}s'.format(time() - tic))
    return mesh


def polydata_smoothing(polydata, lamb=0.95, iterations=500):
    tic = time()
    print('Smoothing... ', end='', flush=True)
    mesh_tmp = trimesh.Trimesh(vertices=polydata.points, faces=polydata.faces.reshape(-1, 4)[:, 1:])
    mesh = trimesh.smoothing.filter_mut_dif_laplacian(mesh_tmp, lamb=lamb, iterations=iterations)
    print('  {:.2f}s'.format(time() - tic))
    return pv.wrap(mesh)


def pyacvd_remeshing(polydata, subdivide=3, cluster=20000):
    tic = time()
    print('Remeshing... ', end='', flush=True)
    clus = pyacvd.Clustering(polydata)
    clus.subdivide(subdivide)
    clus.cluster(cluster)
    remeshed = clus.create_mesh()
    print('  {:.2f}s'.format(time() - tic))
    return remeshed


def convert_triangle_mesh_to_graph(mesh):
    node_list = [list(mesh.points[j]) for j in range(mesh.n_points)]
    node_dict = dict()
    for j, n in enumerate(node_list):
        node_dict[j] = n
    faces = mesh.faces.reshape(-1, 4)
    edges0 = faces[:, 1:3]  # edges AB
    edges1 = faces[:, (1, 3)]  # edges AC
    edges2 = faces[:, 2:]  # edges BC
    edges = np.vstack((edges0, edges1, edges2))
    G = nx.Graph()
    G.add_nodes_from(node_dict)
    G.add_edges_from(edges)
    return G
