import main
import pandas as pd

path = '.'

param = pd.Series({
    "theory": "eft",
    'eftlin': False,
    "alphaB0": -0.24,
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
    "base": f"{path}/compf-24c/",
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
    "Npost_FAS": 10,
    "ncyc": 1,
    "domg": True,
    "epsrel": 1e-2,
    "verbose": 1,
    "evolution_table":'no',
    "Om_lambda":0.742589237,
    })

paramcopy = param.copy(deep=True)

larr = ['a','b','c','d','e']
aarr = [-0.2,-0.1,0,0.1,0.2]

larr = ['e']
aarr = [0.2]

for x in range(len(larr)):
    param = paramcopy.copy(deep=True)
    param['alphaM0'] = aarr[x]
    param['base'] = f"{path}/compf-2-2-24{larr[x]}/"
    print(f'Running case {larr[x]}... {param["alphaM0"]}.')
    main.run(param)

# base case with 5-5
