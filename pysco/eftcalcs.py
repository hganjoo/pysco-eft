"""This module computes a few quantities required for the EFT solver. 

Himanish Ganjoo, 20/11/24

"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import List
from astropy.constants import pc,G

def geteft(
        param: pd.Series,
        tables: List[interp1d]) -> List[np.float32]:
    
    """
    Obtain some derived EFT parameters:
    alphaB, alphaM, C2, C4.
    Commented support for other params like xi,nu,mu_phi
    """

    alphaB0 = param["alphaB0"]
    alphaM0 = param["alphaM0"]
    a = param["aexp"]
    Eval = tables[2] 
    E = Eval(np.log(a)) / param["H0"]
    om_m = param["Om_m"]

    evt = a ** (
            -3 * (1 + param["w0"] + param["wa"])
        ) * np.exp(-3 * param["wa"] * (1 - a))

    om_ma = param['Om_m'] / (param["Om_m"] + param["Om_lambda"] * evt * a ** 3)
    alphaB = alphaB0*(1-om_ma) / (1-om_m)
    alphaM = alphaM0*(1-om_ma) / (1-om_m)
    HdotbyH2 = -1.5*om_ma
    #Ia = np.power(om_ma,param["alphaM0"]/(3 * (1 - om_m)))
    Ia = tables[13](np.log(a))

    et1 = a**( -3*(param["w0"] + param["wa"] ) ) * np.exp(3*param["wa"] * (1-a))
    abdot = om_m * ( 3*et1*param["Om_lambda"]*param["wa"] - 3*et1*param["Om_lambda"]*(param["wa"] + param["w0"])/a)
    abdot = abdot / (et1*param["Om_lambda"] + om_m)**2
    abdot = abdot / (1 - om_m)
    abdot = abdot * alphaB0 * a
    #C2 = -alphaM + alphaB*(1 + alphaM) + (1 + alphaB)*HdotbyH2 + (3*a**3*alphaB0*om_m)/(a**3*(1 - om_m) + om_m)**2 + a**(-3.)*1.5*Ia*om_m/(E**2)
    C2 = -alphaM + alphaB*(1 + alphaM) + (1 + alphaB)*HdotbyH2 + abdot + a**(-3.)*1.5*Ia*om_m/(E**2)
    C4 = -4*alphaB + 2*alphaM
    #xi = alphaB - alphaM
    #nu = -1*C2 -alphaB*(xi - alphaM)
    #mu_chi = xi/nu
    #mu_phi = 1 + xi*xi/nu

    return [alphaB,alphaM,C2,C4]

