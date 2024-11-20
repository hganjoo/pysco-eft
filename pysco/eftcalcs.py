"""This module computes a few quantities required for the EFT solver. 

Himanish Ganjoo, 20/11/24

"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List

def geteft(
        param: pd.Series,
        tables: List[interp1d]) -> List[np.float32]:

    alphaB0 = param["alphaB0"]
    alphaM0 = param["alphaM0"]
    a = param["aexp"]
    Eval = tables[2] 
    E = Eval(a) / param["H0"]

    om_m = param["Om_m"]
    om_ma = om_m / (om_m + (1-om_m)*a**3)
    alphaB = alphaB0*(1-om_ma) / (1-om_m)
    alphaM = alphaM0*(1-om_ma) / (1-om_m)
    HdotbyH2 = -1.5*om_ma
    Ia = np.power(om_ma,-1*param["alphaM0"]/(3 * (1 - om_m)))

    C2 = -alphaM + alphaB*(1 + alphaM) + (1 + alphaB)*HdotbyH2 + (3*a**3*alphaB0*om_m)/(a**3*(1 - om_m) + om_m)**2 + 1.5*Ia*om_m/(E**2)
    C4 = -4*alphaB + 2*alphaM

    return [alphaB,alphaM,C2,C4]
