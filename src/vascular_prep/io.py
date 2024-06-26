from time import time
import pyvista as pv

def read_file(file_path):
    tic = time()
    print('Reading data... ', end='', flush=True) 
    data = pv.read(file_path)
    
    if isinstance(data, pv.ImageData):
        print('ImageData input... ', end='', flush=True)
        #surf = data.threshold([0.5, 1.5]).extract_surface().connectivity(extraction_mode='largest').triangulate()
        surf = data.contour([1]).extract_surface().connectivity(extraction_mode='largest').triangulate()
    
    elif isinstance(data, pv.PolyData):
        print('PolyData input... ', end='', flush=True)
        surf = data.copy()
    else:
        raise ValueError('Wrong file type specified.')

    print('  {:.2f}s'.format(time() - tic))

    return surf


def save_to_file(data, file_path, clear_data=False):
    out = data.copy()
    if clear_data:
        out.clear_data()
    out.save(file_path)
