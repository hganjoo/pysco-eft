import main
import pandas as pd

path = '.'

param = pd.Series({
    "theory": "newton",
    'eftlin': False,
    "alphaB0": -0.48,
    "alphaM0": 0.0,
    "extra":'04_01',
    "nthreads": 6,
    "H0": 72,
    "Om_m": 0.25733,
    "T_cmb": 2.726,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 700,
    "ncoarse": 8,
    "npart": 256**3,
    "z_start": 25,
    "seed": 42,
    "position_ICS": "center",
    "fixed_ICS": False,
    "paired_ICS": False,
    "dealiased_ICS": False,
    "power_spectrum_file": f"{path}/pk_lcdmw7v2.dat",
    "initial_conditions": "2LPT",
    "base": f"{path}/comp48-700-op-2-2/",
    "z_out": "[0]",
    "output_snapshot_format": "HDF5",
    "save_power_spectrum": "no",
    "integrator": "leapfrog",
    "n_reorder": 50,
    "mass_scheme": "TSC",
    "Courant_factor": 1.0,
    "max_aexp_stepping": 10,
    "linear_newton_solver": "multigrid",
    "gradient_stencil_order": 7,
    "Npre": 2,
    "Npost": 1,
    "Npre_FAS": 2,
    "Npost_FAS": 2,
    "ncyc": 1,
    "domg": True,
    "epsrel": 1e-2,
    "verbose": 1,
    "evolution_table":'no',
    "Om_lambda":0.742589237,
    })

main.run(param)

#Npre, Npost = 3,2 for 256^2 | 1,1 for 128^3
# Box = 328

# 700
#3,6 for 128
#3,10 for 256?

# 500
#3,10 for 256

#1000
#2,1 for linear
#3,6 for FAS