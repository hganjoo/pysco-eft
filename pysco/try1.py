import main
import pandas as pd

path = '.'

param = pd.Series({
    "theory": "eft",
    "alphaB0": 0.1,
    "alphaM0": 0.0,
    "extra":'04_01',
    "nthreads": 1,
    "H0": 72,
    "Om_m": 1.,
    "T_cmb": 0.1,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 100,
    "ncoarse": 3,
    "npart": 8**3,
    "z_start": 40,
    "seed": 42,
    "position_ICS": "center",
    "fixed_ICS": False,
    "paired_ICS": False,
    "dealiased_ICS": False,
    "power_spectrum_file": "pk_lcdmw7v2.dat",
    "initial_conditions": "2LPT",
    "base": f"{path}/pt-modp1/",
    "z_out": "[20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.05,0]",
    "output_snapshot_format": "HDF5",
    "save_power_spectrum": "no",
    "integrator": "leapfrog",
    "n_reorder": 50,
    "mass_scheme": "TSC",
    "Courant_factor": 1.0,
    "max_aexp_stepping": 10,
    "linear_newton_solver": "multigrid",
    "gradient_stencil_order": 5,
    "Npre": 5,
    "Npost": 5,
    "epsrel": 1e-2,
    "verbose": 1,
    "evolution_table":'no',
    "Om_lambda":0
    })

main.run(param)