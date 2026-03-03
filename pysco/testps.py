import numpy as np
import matplotlib.pyplot as plt
import pspec
from scipy.interpolate import interp1d

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Modern"
})

import yt

ds = yt.load('/Users/himanishganjoo/Dropbox/prograceray/ecosmog-eft-test/src/rlcdm/output_00011/info_00011.txt')
ad = ds.all_data()
# construct the 3d array of particle positions. Units are set to Mpc/h.
pos = np.array([ad['particle_position_x'].to('Mpc/h'),ad['particle_position_y'].to('Mpc/h'),ad['particle_position_z'].to('Mpc/h')]).transpose().astype(np.float32)

import mesh

mesh.CIC(np.ascontiguousarray(pos, dtype=np.float32),128)