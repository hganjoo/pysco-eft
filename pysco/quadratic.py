"""
This module implements numerical solutions for a quadratic operator in the context of the EFT of gravity,
based on the work by Cusin et al. (2017). 

The EFT model implemented here has two additional params (alphaB, alphaM). 

Himanish Ganjoo - Nov 2024
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import math


@njit(
        ["f8(f8[:,:,::1],f8,i8,i8,i8,f8,f8,f8,f8,f8,f8,f8,f8,f8)"],
        fastmath=True
)
def solution_quadratic_equation(
    pi: npt.NDArray[np.float64],
    b: np.float64,
    x: np.int64,
    y: np.int64,
    z: np.int64,
    h: np.float64,
    C2: np.float64,
    C4: np.float64,
    alphaB: np.float64,
    alphaM: np.float64,
    H: np.float64,
    a: np.float64,
    M: np.float64,
    rhom: np.float64
) -> np.float64:
    
    """Solution of the quadratic equation governing the pi (chi) field \\
    for the EFT parameters. 

    This computes the solution to pi[i,j,k] in terms of the density field and \\
    the neighbours of the cell [i,j,k].

    Parameters
    ----------
    pi : npt.NDArray[np.float64]
         Pi Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float64]
        Density term at [x,y,z]
    x,y,z : np.int16
        3D indices [i,j,k]
    h : np.float64
        Grid size
    C2, C4, alphaB, alphaM : np.float64
        EFT params
    H : np.float64
        Hubble param
    a : np.float64
        scale factor
    M : np.float64
        time-dependent Planck mass
    rhom : np.float64
        matter density


    Returns
    -------
    np.float64
        Solution of the quadratic equation for Pi at location [x,y,z]
    """

    h2 = h**2
    h4 = h2**2
    aH2 = (a*H)**2
    
    pins = pi[-1 + x,y,z] + pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z] + pi[1 + x,y,z]
    
    av = (-6.*C4)/(h4 * aH2)

    lin = (alphaB*(6.*alphaB - 12.*alphaM) + 6.*C2)/h2
    nlin = -8*pins/(h4)
    
    bv = lin - 0.25*C4*nlin/(aH2)

    pins = pi[-1 + x,y,z] + pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z] + pi[1 + x,y,z]

    lin = (
        (a**2*(-0.5*alphaB + 0.5*alphaM )*b*rhom)/M**2 
        + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
    )

    # Coeff of pi^0 in Q2[pi,pi]
    q2offd = -0.125*((pi[x,-1 + y,-1 + z] - pi[x,-1 + y,1 + z] - pi[x,1 + y,-1 + z] + pi[x,1 + y,1 + z])**2 
    - 16.*((pi[x,y,-1 + z] + pi[x,y,1 + z])*(pi[x,-1 + y,z] + pi[x,1 + y,z]) + pi[-1 + x,y,z]*(pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z]) + (pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z])*pi[1 + x,y,z]) 
    + (pi[-1 + x,y,-1 + z] - pi[-1 + x,y,1 + z] - pi[1 + x,y,-1 + z] + pi[1 + x,y,1 + z])**2 
    + (pi[-1 + x,-1 + y,z] - pi[-1 + x,1 + y,z] - pi[1 + x,-1 + y,z] + pi[1 + x,1 + y,z])**2)/(h4)

    cv = lin - 0.25*C4*q2offd/(aH2)

    dsc = np.sqrt(bv**2 - 4*av*cv)
    return (-bv - dsc) / (2*av)


@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float64,
    alphaB: np.float64,
    alphaM: np.float64,
    a: np.float64,
    M: np.float64,
    rhom: np.float64
    
) -> npt.NDArray[np.float32]:
    """
    HG: 14/11/2024

    VERY ROUGH VERSION - have to decide how to initialise the scalar field \\
    this might be too slow
    
    Solution for the Chi field \\
    using the linear order DE solution \\
    Laplacian[Chi] = mu_chi * delta 

    Parameters
    ----------
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, alphaB0, alphaM0 : np.float32
        EFT params, basic and derived, taken from cosmotable
    M: np.float32
        Time-dependent Planck mass (taken from cosmotable)
    a: np.float32
        Scale factor
    rhom: np.float32
        matter density at current time

    Returns
    -------
    npt.NDArray[np.float32]
        Chi field

    
    """


    xi = alphaB - alphaM
    nu = -C2 - alphaB*(xi - alphaM)
    mu_chi = xi/nu
    
    pi = np.empty_like(b)
    ncells_1d = b.shape[0]
    for i in range(ncells_1d):
        for j in range(ncells_1d):
            for k in range(ncells_1d):
                pi[i, j, k] = (1./6)*((pi[-1 + i,j,k] + pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k] + pi[1 + i,j,k]) 
                                      - h*h*0.5*a*a*mu_chi*rhom*b[i,j,k]/(M*M))
    return pi


@njit(
    ["void(f8[:,:,::1], f8[:,:,::1], f8, f8, f8, f8, f8, f8, f8, f8, f8)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel(
    x: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    h: np.float64,
    C2: np.float64,
    C4: np.float64,
    alphaB: np.float64,
    alphaM: np.float64,
    H: np.float64,
    a: np.float64,
    M: np.float64,
    rhom: np.float64
) -> None:
    """Gauss-Seidel quadratic equation solver for Pi (Chi) \\
    Solve the roots of u in the equation: \\
    au^2 + bu + c = 0 \\
    in Simple EFT, Cusin et al (2017)\\

    Parameters
    ----------
    x : npt.NDArray[np.float64]
        Chi Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float64]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float64
        Grid size
    C2, C4, alphaB, alphaM : np.float64
        EFT params
    H : np.float64
        Hubble param
    a : np.float64
        scale factor
    M : np.float64
        time-dependent Planck mass
    rhom : np.float64
        matter density

    """
    half_ncells_1d = x.shape[0] >> 1
    # Computation Red
    for i in prange(x.shape[0] >> 1):
        ii = 2 * i
        iim1 = ii - 1
        for j in prange(half_ncells_1d):
            jj = 2 * j
            jjm1 = jj - 1
            for k in prange(half_ncells_1d):
                kk = 2 * k
                kkm1 = kk - 1

                x[iim1, jjm1, kkm1] = solution_quadratic_equation(x,b[iim1,jjm1,kkm1],iim1,jjm1,kkm1,h,C2,C4,alphaB,alphaM,H,a,M,rhom)
                x[iim1, jj, kk] = solution_quadratic_equation(x,b[iim1,jj,kk],iim1,jj,kk,h,C2,C4,alphaB,alphaM,H,a,M,rhom)
                x[ii, jjm1, kk] = solution_quadratic_equation(x,b[ii,jjm1,kk],ii,jjm1,kk,h,C2,C4,alphaB,alphaM,H,a,M,rhom)
                x[ii, jj, kkm1] = solution_quadratic_equation(x,b[ii,jj,kkm1],ii,jj,kkm1,h,C2,C4,alphaB,alphaM,H,a,M,rhom)

    # Computation Black
    for i in prange(half_ncells_1d):
        ii = 2 * i
        iim1 = ii - 1
        for j in prange(half_ncells_1d):
            jj = 2 * j
            jjm1 = jj - 1
            for k in prange(half_ncells_1d):
                kk = 2 * k
                kkm1 = kk - 1

                x[iim1, jjm1, kk] = solution_quadratic_equation(x,b[iim1,jjm1,kk],iim1,jjm1,kk,h,C2,C4,alphaB,alphaM,H,a,M,rhom)
                x[iim1, jj, kkm1] = solution_quadratic_equation(x,b[iim1,jj,kkm1],iim1,jj,kkm1,h,C2,C4,alphaB,alphaM,H,a,M,rhom)
                x[ii, jjm1, kkm1] = solution_quadratic_equation(x,b[ii,jjm1,kkm1],ii,jjm1,kkm1,h,C2,C4,alphaB,alphaM,H,a,M,rhom)
                x[ii, jj, kk] = solution_quadratic_equation(x,b[ii,jj,kk],ii,jj,kk,h,C2,C4,alphaB,alphaM,H,a,M,rhom)





