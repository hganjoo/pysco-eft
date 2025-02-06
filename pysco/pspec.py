import h5py
import mesh
import fourier
import numpy as np

def get_pspec(file_path):

    with h5py.File(file_path, "r") as hdf_file:
   
        position = hdf_file['position'][:]
        boxlen = hdf_file.attrs['boxlen']
        ncells_1d = 2**(hdf_file.attrs['ncoarse'])
        density = mesh.CIC(position, ncells_1d)
        aval = hdf_file.attrs['aexp']

    density_fourier = fourier.fft_3D_real(density,1)
    k, Pk, Nmodes = fourier.fourier_grid_to_Pk(density_fourier, 2)
    Pk *= (boxlen / len(density) ** 2) ** 3
    k *= 2 * np.pi / boxlen

    return k,Pk,aval



