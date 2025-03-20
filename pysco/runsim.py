from pathlib import Path
import main

path = '.'

param = {
    "nthreads": 8,
    "theory": "eft",
    'eftlin': False,
    "alphaB0": -0.24,
    "alphaM0": 0.0,
    "extra":'04_01',
    "H0": 72,
    "Om_m": 0.25733,
    "T_cmb": 2.726,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 328,
    "ncoarse": 8,
    "npart": 256**3,
    "z_start": 25,
    "seed": 42,
    "position_ICS": "center",
    "fixed_ICS": False,
    "paired_ICS": False,
    "dealiased_ICS": False,
    "power_spectrum_file": f"{path}/pt_ps.dat",
    "initial_conditions": "3LPT",
    "base": f"{path}/comp24/",
    "z_out": "[0]",
    "output_snapshot_format": "HDF5",
    "save_power_spectrum": "no",
    "integrator": "leapfrog",
    "n_reorder": 50,
    "mass_scheme": "TSC",
    "Courant_factor": 1.0,
    "max_aexp_stepping": 10,
    "linear_newton_solver": "multigrid",
    "gradient_stencil_order": 5,
    "Npre": 2,
    "Npost": 1,
    "domg": True,
    "epsrel": 1e-2,
    "verbose": 1,
    "evolution_table":'no',
    "Om_lambda":0.742589237,
    }

print('Run.',param['nthreads'])

main.run(param)
