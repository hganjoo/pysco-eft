import numpy as np
import main
import pandas as pd

path = './alphaMtest/'

param = pd.Series({
    "theory": "eft",
    'eftlin': False,
    "alphaB0": -0.48,
    "extra":'04_01',
    "nthreads": 6,
    "H0": 72,
    "Om_m": 0.25733,
    "T_cmb": 2.726,
    "N_eff": 3.044,
    "w0": -1.0,
    "wa": 0.0,
    "boxlen": 656,
    "ncoarse": 7,
    "npart": 128**3,
    "z_start": 25,
    "seed": 42,
    "position_ICS": "center",
    "fixed_ICS": False,
    "paired_ICS": False,
    "dealiased_ICS": False,
    "power_spectrum_file": f"./pk_lcdmw7v2.dat",
    "initial_conditions": "2LPT",
    #"base": f"{path}/test-alpham7/",
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
    "Npre": 3,
    "Npost": 2,
    "Npre_FAS": 5,
    "Npost_FAS": 5,
    "ncyc": 1,
    "domg": True,
    "epsrel": 1e-2,
    "verbose": 1,
    "evolution_table":'no',
    "Om_lambda":0.742589237,
    })

abs = np.arange(-0.4,0.01,0.04)
print(abs)

for alphaB in abs:
    print(alphaB)
    pthis = param.copy()
    pthis['alphaM0'] = alphaB
    pthis['base'] = f"{path}/{alphaB}/"
    main.run(pthis)

    pthis = param.copy()
    pthis['alphaM0'] = alphaB
    pthis['base'] = f"{path}/{alphaB}/"
    pthis['eftlin'] = True
    main.run(pthis)

'''pthis = param.copy()
pthis['alphaB0'] = -0.48
pthis['theory'] = 'newton'
pthis['base'] = f"{path}/-0.48/"
main.run(pthis)'''

# alphaBtest: alphaM = 0, alphaB varies from -0.48 to -0.04
# alphaMtest: alphaB = -0.48, alphaM varies from -0.4 to 0
    

