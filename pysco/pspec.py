import h5py
import mesh
import fourier
import numpy as np
import yt

def get_pspec(file_path):

    with h5py.File(file_path, "r") as hdf_file:
   
        position = hdf_file['position'][:]
        print(position.shape,position.dtype)
        boxlen = hdf_file.attrs['boxlen']
        ncells_1d = 2**(hdf_file.attrs['ncoarse'])
        density = mesh.CIC(position, ncells_1d)
        aval = hdf_file.attrs['aexp']

    density_fourier = fourier.fft_3D_real(density,1)
    k, Pk, Nmodes = fourier.fourier_grid_to_Pk(density_fourier, 2)
    Pk *= (boxlen / len(density) ** 2) ** 3
    k *= 2 * np.pi / boxlen

    return k,Pk,aval

def get_ram(file_path):

    ds = yt.load(file_path)
    ad = ds.all_data()
    # construct the 3d array of particle positions. Units are set to Mpc/h.
    position = np.array([ad['particle_position_x'].to('Mpc/h'),ad['particle_position_y'].to('Mpc/h'),ad['particle_position_z'].to('Mpc/h')]).transpose().astype(np.float32)
    boxlen = ds.domain_width.to('Mpc/h')[0]
    ncells_1d = ds.domain_dimensions[0].astype(np.int32)
    aval = ds.scale_factor
    density = mesh.CIC(position, ncells_1d)
    density_fourier = fourier.fft_3D_real(density,1)
    k, Pk, Nmodes = fourier.fourier_grid_to_Pk(density_fourier, 2)
    Pk *= (boxlen / len(density) ** 2) ** 3
    k *= 2 * np.pi / boxlen

    return k,Pk,aval


    



