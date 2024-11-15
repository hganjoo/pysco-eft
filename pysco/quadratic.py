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
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def operator(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    M: np.float32,
    rhom: np.float32) -> npt.NDArray[np.float32]:
    """Quadratic operator

    a pi^2 + b pi + c = 0 \\
    EFT from Cusin et al. (2017)\\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Hubble param
    a : np.float32
        scale factor
    M : np.float32
        time-dependent Planck mass
    rhom : np.float32
        matter density
        
    Returns
    -------
    npt.NDArray[np.float32]
        Quadratic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    """
    ncells_1d = pi.shape[0]
    result = np.empty_like(pi)
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):

                h2 = h**2
                h4 = h2**2
                aH2 = (a*H)**2
                
                pins = pi[-1 + i,j,k] + pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k] + pi[1 + i,j,k]
                
                av = (-6.*C4)/(h4 * aH2)

                lin = (alphaB*(6.*alphaB - 12.*alphaM) + 6.*C2)/h2
                nlin = -8*pins/(h4)
                
                bv = lin - 0.25*C4*nlin/(aH2)

                lin = (
                    (a**2*(-0.5*alphaB + 0.5*alphaM )*b*rhom)/M**2 
                    + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
                )

                # Coeff of pi^0 in Q2[pi,pi]
                q2offd = -0.125*((pi[i,-1 + j,-1 + k] - pi[i,-1 + j,1 + k] - pi[i,1 + j,-1 + k] + pi[i,1 + j,1 + k])**2 
                - 16.*((pi[i,j,-1 + k] + pi[i,j,1 + k])*(pi[i,-1 + j,k] + pi[i,1 + j,k]) + pi[-1 + i,j,k]*(pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k]) + (pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k])*pi[1 + i,j,k]) 
                + (pi[-1 + i,j,-1 + k] - pi[-1 + i,j,1 + k] - pi[1 + i,j,-1 + k] + pi[1 + i,j,1 + k])**2 
                + (pi[-1 + i,-1 + j,k] - pi[-1 + i,1 + j,k] - pi[1 + i,-1 + j,k] + pi[1 + i,1 + j,k])**2)/(h4)

                cv = lin - 0.25*C4*q2offd/(aH2)

                result[i,j,k] = av*pi[i,j,k]**2 + bv*pi[i,j,k] + cv
    
    return result



@njit(
        ["f4(f4[:,:,::1],f4,i4,i4,i4,f4,f4,f4,f4,f4,f4,f4,f4,f4)"],
        fastmath=True
)
def solution_quadratic_equation(
    pi: npt.NDArray[np.float32],
    b: np.float32,
    x: np.int32,
    y: np.int32,
    z: np.int32,
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    M: np.float32,
    rhom: np.float32
) -> np.float32:
    
    """Solution of the quadratic equation governing the pi (chi) field \\
    for the EFT parameters. 

    This computes the solution to pi[i,j,k] in terms of the density field and \\
    the neighbours of the cell [i,j,k].

    Parameters
    ----------
    pi : npt.NDArray[np.float32]
         Pi Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term at [x,y,z]
    x,y,z : np.int16
        3D indices [i,j,k]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Hubble param
    a : np.float32
        scale factor
    M : np.float32
        time-dependent Planck mass
    rhom : np.float32
        matter density


    Returns
    -------
    np.float32
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
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def gauss_seidel(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    M: np.float32,
    rhom: np.float32) -> npt.NDArray[np.float32]:
    """Gauss-Seidel quadratic equation solver \\
    Solve the roots of u in the equation: \\
    a u^2 + bu + c = 0 \\
    for the EFT in Cusin et al (2017)\\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Hubble param
    a : np.float32
        scale factor
    M : np.float32
        time-dependent Planck mass
    rhom : np.float32
        matter density
        
    """

    ncells_1d = pi.shape[0]

    for ix in range(-1,ncells_1d - 1):
            for iy in range(-1,ncells_1d - 1):
                for iz in range(-1,ncells_1d - 1):
                    pi[ix,iy,iz] = solution_quadratic_equation(pi,b[ix,iy,iz],ix,iy,iz,h,C2,C4,alphaB,alphaM,H,a,M,rhom)
                    



@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    a: np.float32,
    M: np.float32,
    rhom: np.float32
    
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


