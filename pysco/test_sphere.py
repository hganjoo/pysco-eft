import numpy as np
import pandas as pd
from solver import fft
from multigrid import V_cycle, F_cycle, W_cycle
import eftcalcs,cosmotable
from utils import prod_vector_scalar
from mesh import (
    TSC,
    CIC,
    derivative5,
    derivative7,
    invTSC_vec,
    invCIC_vec,
    invTSC,
    invCIC,
)
from utils import linear_operator_inplace
#import ewald
import matplotlib.pyplot as plt
from utils import set_units
from astropy.constants import G, pc, c
#import hernquist
from scipy.special import erfc
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "Serif"
plt.rcParams["xtick.top"] = "True"
matplotlib.rcParams.update({"font.size": 16})
plt.rcParams["xtick.top"] = "True"


def force_point_mass(r):
    return 1.0 / r**2


def potential_point_mass(r):
    return -1.0 / r


def potential_sphere(r, R):
    inside_sphere = r < R
    potential = np.zeros_like(r)
    potential[inside_sphere] = -(3 * R**2 - r[inside_sphere] ** 2) / (2 * R**3)
    potential[~inside_sphere] = -1.0 / r[~inside_sphere]
    return potential


def force_sphere(r, R):
    inside_sphere = r < R
    force = np.zeros_like(r)
    force[inside_sphere] = r[inside_sphere] / R**3
    force[~inside_sphere] = 1.0 / r[~inside_sphere] ** 2
    return force


def force_sphere_periodic(x, y, z, R):
    dmax = 8
    res = np.zeros_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    nx = x / norm
    ny = y / norm
    nz = z / norm
    norm = 0
    for ix in range(-dmax, dmax + 1):
        for iy in range(-dmax, dmax + 1):
            for iz in range(-dmax, dmax + 1):
                rx = x + ix
                ry = y + iy
                rz = z + iz
                r2 = rx**2 + ry**2 + rz**2
                mask = r2 < 2.6**2
                res[mask] += (
                    force_sphere(np.sqrt(r2[mask]), R)
                    * (rx[mask] * nx[mask] + ry[mask] * ny[mask] + rz[mask] * nz[mask])
                    / np.sqrt(r2[mask])
                )
    return res


def force_sphere_periodic_full(x, y, z, R):
    alpha = 2
    dmax = 4
    res = np.zeros_like(x)
    norm = np.sqrt(x * x + y * y + z * z)
    nx = x / norm
    ny = y / norm
    nz = z / norm
    norm = 0
    for ix in range(-dmax, dmax + 1):
        for iy in range(-dmax, dmax + 1):
            for iz in range(-dmax, dmax + 1):
                rx = x + ix
                ry = y + iy
                rz = z + iz
                r2 = rx**2 + ry**2 + rz**2
                mask = r2 < 2.6**2
                res[mask] += (
                    force_sphere(np.sqrt(r2[mask]), R)
                    * (rx[mask] * nx[mask] + ry[mask] * ny[mask] + rz[mask] * nz[mask])
                    / np.sqrt(r2[mask])
                    * (
                        erfc(alpha * np.sqrt(r2[mask]))
                        + 2 * alpha / np.sqrt(np.pi) * np.exp(-(alpha**2) * r2[mask])
                    )
                )

    for ix in range(-dmax, dmax + 1):
        for iy in range(-dmax, dmax + 1):
            for iz in range(-dmax, dmax + 1):
                rx = x + ix
                ry = y + iy
                rz = z + iz
                r2 = rx**2 + ry**2 + rz**2
                if ix == 0 and iy == 0 and iz == 0:
                    continue
                mask = (ix**2 + iy**2 + iz**2) < 8
                res[mask] += (
                    -2
                    * (ix * nx[mask] + iy * ny[mask] + iz * nz[mask])
                    / (ix**2 + iy**2 + iz**2)
                    * np.exp(-np.pi**2 * (ix**2 + iy**2 + iz**2) / alpha**2)
                    * np.sin(
                        2 * np.pi * (ix * rx[mask] + iy * ry[mask] + iz * rz[mask])
                    )
                )

    return res / np.pi


param = pd.Series(
    {
        "theory":"newton",
        "nthreads": 4,
        "boxlen": 100.0,
        "npart": 128**3,
        "T_cmb": 2.726,
        "compute_additional_field": False,
        "ncoarse": 7,
        "Npre": 1,
        "Npost": 1,
        "Npre_FAS": 2,
        "Npost_FAS": 2,
        "alphaB0": -0.24,
        "alphaM0": 0.0,
        "aexp": 1.0,
        "H0": 70,
        "Om_m": 0.3,
        "Om_l": 0.7,
        "MAS_index": 0,
        "linear_newton_solver": "multigrid",
        "N_eff": 3.044,
        "w0":-1.0,
        "wa": 0.0,
        "base": './sphtest/',
        "extra": 'spht',
        "eftlin": False
    }
)


ncells_1d = 2 ** int(param["ncoarse"])
ncells = ncells_1d
h = np.float32(1.0 / ncells_1d)
#
set_units(param)
g = G.value * 1e-9  # m3/kg/s2 -> km3/kg/s2
GM = g * param["mpart"]

tables = cosmotable.generate(param)

alphaB,alphaM,C2,C4,mu_phi,Ia,xi,nu = eftcalcs.geteft(param,tables)
eft_quantities = eftcalcs.geteft(param,tables)
param["alphaB"] = eft_quantities[0]
param["alphaM"] = eft_quantities[1]
param["C2"] = eft_quantities[2]
param["C4"] = eft_quantities[3]
Eval = tables[2] 
param["H"] = Eval(np.log(param["aexp"])) / param["H0"]

h = np.float32(1.0 / ncells_1d)
# pos_point_mass = np.random.rand(3).astype(np.float32)
pos_point_mass_edge = np.array([0.5, 0.5, 0.5]).astype(np.float32)
pos_point_mass_center = np.array([0.5 + 0.5 * h, 0.5 + 0.5 * h, 0.5 + 0.5 * h]).astype(
    np.float32
)
'''pos_test_particles = (
    0.3 * (np.random.rand(128**3, 3) - 0.5) + pos_point_mass_edge
).astype(np.float32)

derivative = derivative7'''

rhs = CIC(np.array([pos_point_mass_center]), ncells_1d)
rhs *= 1.5 * param["Om_m"]
minus_one_sixth_h2 = np.float32(-(h**2) / 6)
potential_V = prod_vector_scalar(rhs, minus_one_sixth_h2)
V_cycle(potential_V, rhs, param)
V_cycle(potential_V, rhs, param)
V_cycle(potential_V, rhs, param)



#plt.imshow(potential_V[:,:,32])
#plt.show()

nc = int(ncells/2) - 0.5
cent = int(ncells/2)

rv = np.indices((ncells,ncells,ncells)).astype(np.float32) - nc
rv = h*np.sqrt(np.sum(rv**2,axis=0))
norm = param["unit_l"] / param["unit_t"] ** 2 / (GM / param["unit_l"] ** 2)

xv = h*(np.arange(ncells) + 0.5)
pia = potential_point_mass(rv)
pix = np.array([np.mean(potential_V[i,:,:]) for i in range(ncells)])
piax = np.array([np.mean(pia[i,:,:]) for i in range(ncells)])
plt.plot(xv,norm*pix)
plt.plot(xv,piax)
plt.show()