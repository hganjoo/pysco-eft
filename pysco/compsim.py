import main
import pandas as pd

path = '/Users/himanishganjoo/Documents/rome-results/pysco_comp/'

param = pd.Series({
    "theory": "newton",
    'eftlin': False,
    "alphaB0": -0.248458,
    "alphaM0": -0.156236,
    "extra":'04_01',
    "nthreads": 6,
    "H0": 0.644724*100,
    "Om_m": 0.377719,
    "T_cmb": 2.726,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 328.125,
    "ncoarse": 9,
    "npart": 512**3,
    "z_start": 37.31975999924061,
    "seed": 42,
    "position_ICS": "center",
    "fixed_ICS": False,
    "paired_ICS": False,
    "dealiased_ICS": False,
    "power_spectrum_file": f"{path}/pk_lcdmw7v2.dat",
    "initial_conditions": path + "/new_ics.hdf5",
    "base": path,
    #"z_out": "[2.00867,0.9861,0.07864868584040763,-0.05486144090248224]",
    "z_out": "[2,1,0]",
    "output_snapshot_format": "HDF5",
    "save_power_spectrum": "no",
    "integrator": "leapfrog",
    "n_reorder": 50,
    "mass_scheme": "CIC",
    "Courant_factor": 1.0,
    "max_aexp_stepping": 10,
    "linear_newton_solver": "multigrid",
    "gradient_stencil_order": 5,
    "Npre": 3,
    "Npost": 2,
    "Npre_FAS": 5,
    "Npost_FAS": 5,
    "ncyc": 1,
    "domg": True,
    "epsrel": 1e-2,
    "verbose": 1,
    "evolution_table":'no',
    "Om_lambda":1 - 0.377719,
    "parametrized_mu0":1
    })

main.run(param)

#Npre, Npost = 3,2 for 256^2 | 1,1 for 128^3
# Box = 328