import os
import sys
import numpy as np
import pyvista as pv
import pymeshfix as mf
import pyacvd
from time import time
import trimesh
import networkx as nx
import meshio
import subprocess
from vmtk import vmtkscripts


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


def mmg_remesh(input_mesh, hausd=0.3, hmax=2, hmin=1.5, max_aspect_ratio=None, max_iter=3, verbose=False):
    mesh_id = os.getpid()
    input_mesh.clear_data()
    pv.save_meshio(f'{mesh_id}.mesh', input_mesh)

    subprocess.run([sys.executable, '-m', 'mmgs',
                f'{mesh_id}.mesh',
                '-hausd', str(hausd),
                '-hmax', str(hmax),
                '-hmin', str(hmin),
                '-nr',
                '-nreg',
                '-xreg'
                '-optim',
                f'{mesh_id}_remeshed.mesh'], stdout=subprocess.DEVNULL)

    new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
    pvmesh = pv.utilities.from_meshio(new_mesh)

    if max_aspect_ratio is not None:
        qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')
        it = 0
        while np.max(qual['CellQuality']) > max_aspect_ratio and it < max_iter:
            it += 1

            subprocess.run([sys.executable, '-m', 'mmgs',
            f'{mesh_id}.mesh',
            '-hausd', str(hausd * 2),
            '-hmax', str(hmax),
            '-hmin', str(hmin),
            '-nr',
            '-nreg',
            '-xreg'
            '-optim',
            f'{mesh_id}_remeshed.mesh'], stdout=subprocess.DEVNULL)

            new_mesh = meshio.read(f'{mesh_id}_remeshed.mesh')
            pvmesh = pv.utilities.from_meshio(new_mesh)
            qual = pvmesh.compute_cell_quality(quality_measure='aspect_ratio')

            if verbose: print('Max aspect ratio:', np.max(qual['CellQuality']))

    else:
        if verbose: print('Max aspect ratio:', np.max(qual['CellQuality']))

    os.remove(f'{mesh_id}.mesh')
    os.remove(f'{mesh_id}_remeshed.mesh')
    os.remove(f'{mesh_id}_remeshed.sol')

    return pvmesh.extract_surface()


def extract_centerline(surface, sourceCoords, targetCoords, resampling=0.05, appendEndPoints=0):
    cl_filter = vmtkscripts.vmtkCenterlines()
    cl_filter.Surface = surface
    cl_filter.SeedSelectorName = "pointlist"
    cl_filter.SourcePoints = sourceCoords
    cl_filter.TargetPoints = targetCoords
    #cl_filter.CapDisplacement = 1
    cl_filter.AppendEndPoints = appendEndPoints
    cl_filter.Resampling = 1
    cl_filter.ResamplingStepLength = resampling
    cl_filter.Execute()

    attr = vmtkscripts.vmtkCenterlineAttributes()
    attr.Centerlines = cl_filter.Centerlines
    attr.Execute()

    geo = vmtkscripts.vmtkCenterlineGeometry()
    geo.Centerlines = attr.Centerlines
    geo.LineSmoothing = 0
    geo.OutputSmoothingLines = 0
    geo.Execute()
    cl = geo.Centerlines

    return pv.wrap(cl)
