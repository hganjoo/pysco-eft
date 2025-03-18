# imports

import pandas as pd
import eftcalcs
import cosmotable
import utils
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import prange
from scipy.integrate import cumulative_trapezoid as cumtrapz
from astropy.constants import G,pc

import solver
import multigrid
import quadratic
import laplacian


from scipy.stats import binned_statistic as bst
def getfr(arr,h,nc):
    # Computes radial component of gradient of a field relative to grid center
    grad = np.gradient(arr) # 3D gradient, is a list with 3 elements. Each element is the numerical diff in one dimension.
    sh = arr.shape[0]
    x,y,z = h*(np.indices((sh,sh,sh)) - nc) # coordinates of each point in the grid relative to the center, stored in x,y,z
    v = ((x/rv)*grad[0] + (y/rv)*grad[1] + (z/rv)*grad[2])/h # (x/rs)i + (y/rs)j + (z/rs)k is the unit vector in the radial dir.
    return v

import numpy as np

def hist(x, values, bins):
    """
    A fast alternative to scipy.stats.binned_statistic() for computing the mean of values in bins.
    
    Parameters:
        x (array-like): The input data to be binned.
        values (array-like): The values associated with x, used to compute the mean in each bin.
        bins (array-like): The bin edges.
        
    Returns:
        bin_means (ndarray): The mean values in each bin.
        bin_edges (ndarray): The edges of the bins.
    """
    x = np.asarray(x).flatten()
    values = np.asarray(values).flatten()
    bin_edges = np.asarray(bins)

    # Compute bin indices
    bin_indices = np.digitize(x, bin_edges) - 1  # Adjust for zero-based indexing

    # Remove out-of-bounds indices
    valid = (bin_indices >= 0) & (bin_indices < len(bin_edges) - 1)
    bin_indices = bin_indices[valid]
    values = values[valid]

    # Compute bin counts and sums
    bin_counts = np.bincount(bin_indices, minlength=len(bin_edges) - 1)
    bin_sums = np.bincount(bin_indices, weights=values, minlength=len(bin_edges) - 1)

    # Compute bin means (handling empty bins)
    bin_means = np.divide(bin_sums, bin_counts, out=np.full_like(bin_sums, np.nan, dtype=float), where=bin_counts > 0)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return bin_means, bin_centers

# params, utils, EFT params

path = '.'

param = pd.Series({
    "aexp": 1.,
    "theory": "eft",
    "base": path,
    "extra": 'nil',
    "alphaB0": -0.1,
    "alphaM0": 0.3,
    "nthreads": 1,
    "H0": 72,
    "Om_m": 0.3,
    "Om_lambda": 0.7,
    "T_cmb": 2.726,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 100,
    "Npre": 2,
    "Npost": 2,
    "domg": True,
    "epsrel": 1e-2,
    "verbose": 1,
    'newton': False,
    "z_start": 50,
    "compute_additional_field":True,
    "eftlin": False,
    "nsteps":1,
    "verbose":1
    })

# npre, npost = 10,2; without rhs terms
# alphas = -0.5, 0.1

# density profile assignment: isothermal


ncells = 64
param['ncoarse'] = int(np.log(ncells)/np.log(2))
param['npart'] = ncells**3

tables = cosmotable.generate(param)
utils.set_units(param)

eft_quantities = eftcalcs.geteft(param,tables)
param["alphaB"] = eft_quantities[0]
param["alphaM"] = eft_quantities[1]
param["C2"] = eft_quantities[2]
param["C4"] = eft_quantities[3]
Eval = tables[2] 
param["H"] = Eval(np.log(param["aexp"])) / param["H0"]


a = param['aexp']
g = G.value * 1e-9  # m3/kg/s2 -> km3/kg/s2
g = g * param["unit_d"] * param["unit_t"]**2 # g from SI to BU
M = 1./np.sqrt(8*np.pi*g) # g is modified 


density = np.zeros((ncells,ncells,ncells),dtype=np.float32)
h = 1./ncells
mc = 1e-3
rc = 2*h

rs = np.logspace(4,-3,200)*h


def rho_int(r):
    return np.where(r > rc, 0.0, mc / (4 * np.pi * rc * r * r))


H = param['H']
xi = param['alphaB'] - param['alphaM']
nu = -param['C2'] - param['alphaB']*(xi - param['alphaM'])

mpc_to_km = 1e3 * pc.value  #   Mpc -> km
H = param["H0"] / mpc_to_km # H to SI
H = H * param["unit_t"] # From SI to BU
H = H*param['H']


mu_chi = xi/nu
mu_psi = 1 + xi*param['alphaB']/nu
mu_phi = 1 + xi*xi/nu

A = g*mc/(rs**3)


H = param['H']

print(r'Mu values: Chi = {:.2f}, Phi = {:.2f}, Psi = {:.2f}'.format(mu_chi,mu_phi,mu_psi))


x = (nu - np.sqrt(nu*nu - 2*A*param['C4']*xi)) / param['C4']
y = A + (param['alphaB'] - param['alphaM'])*x
z = A + param['alphaB']*x



dchi_dr = x*a*a*H*H*rs
dphi_dr = y*a*a*H*H*rs
dpsi_dr = z*a*a*H*H*rs

chi = cumtrapz(y = dchi_dr,x=rs)
phi = cumtrapz(y = dphi_dr,x=rs)
psi = cumtrapz(y = dpsi_dr,x=rs)

dchi_dr = interp1d(rs,dchi_dr,fill_value='extrapolate')
dphi_dr = interp1d(rs,dphi_dr)
dpsi_dr = interp1d(rs,dpsi_dr)

chi_as = interp1d(rs[:-1],chi,fill_value='extrapolate')
phi_as = interp1d(rs[:-1],phi,fill_value='extrapolate')
psi_as = interp1d(rs[:-1],psi,fill_value='extrapolate')

print('EFT done.')

nc = int(ncells/2) - 0.5
cent = int(ncells/2)

rv = np.indices((ncells,ncells,ncells)).astype(np.float32) - nc
rv = h*np.sqrt(np.sum(rv**2,axis=0))

density[cent-1:cent+1,cent-1:cent+1,cent-1:cent+1] = mc/np.power(2*h,3)

shell_thickness = 1
r_outer = (nc)*h  # Maximum possible radius in the cubic box
r_inner = r_outer - shell_thickness*h  # Inner radius of the shell
shell_mask = (rv >= r_inner) & (rv <= r_outer)
num_voxels = np.sum(shell_mask)
if num_voxels == 0:
    raise ValueError("Shell is too thin or box is too small to contain a shell.")
mcomp = -mc
density[shell_mask] += mcomp / (num_voxels*h**3)

print('Density done. Solving fields...')

# additional field

chi = np.zeros_like(density)

chid = multigrid.FAS(chi,4*np.pi*g*density,h,param)
#quadratic.smoothing(chi,4*3.14*g*density,h,param['C2'],param['C4'],param['alphaB'],param['alphaM'],1,1,5000)

print('Chi done. Solving Phi...')

# gravity

param['compute_additional_field'] = False

pot = np.zeros_like(density)

density = 4*np.pi*g*density + (param['alphaB'] - param['alphaM'])*laplacian.operator(chid,h)

pot = multigrid.linear(pot,density,h,param)

print('Phi done. Plotting...')


# plots
bins = np.logspace(np.log10(h),np.log10(rv.max()),60)

f, ax = plt.subplots(2, 2, gridspec_kw={'height_ratios': [4, 1]})

# First plot (dchi/dr)
plt.sca(ax[0, 0])
bes, p,= hist(rv.flatten(), getfr(chid, h, nc).flatten(), bins=bins)
plt.loglog(p / h, -bes, 'o', markersize=3)

bea, p,= hist(rv.flatten(), dchi_dr(rv.flatten()), bins=bins)
plt.loglog(p / h, -bea, c='k', lw=1.5, alpha=0.5)

bel = -mu_chi * g * mc / (p * p)

plt.loglog(p / h, bel , ls='dotted')
plt.xlim(2, nc)
plt.legend(['Solution', 'Analytical', 'Linear'])
plt.ylim(1e-4, 1e-1)
plt.title(r'$-\nabla_r \chi$')

# First error plot
plt.sca(ax[1, 0])
pdf = -(-bea - bes) / bea
plt.semilogx(p / h, pdf, '.', c='k')
plt.xlabel('$r$ / GridSize')
plt.ylabel('Rel. Error')
plt.xlim(2, nc)
plt.ylim(-0.1, 0.1)
plt.axhline(0, c='grey', ls='dotted', lw=1)
plt.axhline(0.05, c='grey', ls='dotted', lw=1)
plt.axhline(-0.05, c='grey', ls='dotted', lw=1)

print('Err mean:',np.nanmean(pdf))

# Second plot (dphi/dr)
plt.sca(ax[0, 1])
bes, p,= hist(rv.flatten(), getfr(pot, h, nc).flatten(), bins=bins)
plt.loglog(p / h, bes, 'o', markersize=3)

bea, p,= hist(rv.flatten(), dchi_dr(rv.flatten()), bins=bins)
plt.loglog(p / h, bea, c='k', lw=1.5, alpha=0.5)

bel = mu_phi * g * mc / (p * p)

plt.loglog(p / h, bel , ls='dotted')
plt.xlim(2, nc)
plt.legend(['Solution', 'Analytical', 'Linear'])
plt.ylim(1e-4, 1)
plt.title(r'$\nabla_r \Phi$')

# Second error plot
plt.sca(ax[1, 1])
pdf = -(bea - bes) / bea
plt.semilogx(p / h, pdf, '.', c='k')
plt.xlabel('$r$ / GridSize')
#plt.ylabel('Rel. Error')
plt.xlim(2, nc)
plt.ylim(-0.1, 0.1)
plt.axhline(0, c='grey', ls='dotted', lw=1)
plt.axhline(0.05, c='grey', ls='dotted', lw=1)
plt.axhline(-0.05, c='grey', ls='dotted', lw=1)

# Set suptitle
f.suptitle('Nonlinear EFT, Cubic Screening\nPoint Mass in a Box, ${}^3$ cells\n'.format(ncells) +
           r'$\alpha_{{B0}} = {:.1f}$, $\alpha_{{M0}} = {:.1f}$, $a = 1$'.format(param['alphaB'], param['alphaM']))

plt.tight_layout(rect=[0, 0, 1, 0.99])  # Adjust layout to fit suptitle

plt.show()



#plt.savefig('potcomp_resrhs.png',dpi=500)

'''bins = np.logspace(np.log10(h),np.log10(rv.max()),60)

bes, p,= hist(rv.flatten(), getfr(chid, h, nc).flatten(), bins=bins)
plt.semilogx(p / h, -bes, 'o', markersize=3)
#plt.yscale('symlog')
plt.show()'''

