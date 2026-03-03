import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh
import laplacian
import utils

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
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
    a: np.float32
) -> npt.NDArray[np.float32]:
    
    p = alphaB * (2. * alphaM - alphaB) - C2
    q = (alphaM - alphaB)
    r = -0.25 * C4 / (a * a * H)**2
    
    ncells_1d = pi.shape[0]
    
    divF = np.zeros_like(pi)
    

    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):

                # ===== Fx at i+1/2 =====
                pi_x_p = (pi[i+1, j, k] - pi[i, j, k]) / h
                pi_y_p = 0.25 * ((pi[i, j+1, k] - pi[i, j-1, k]) + (pi[i+1, j+1, k] - pi[i+1, j-1, k])) / h
                pi_z_p = 0.25 * ((pi[i, j, k+1] - pi[i, j, k-1]) + (pi[i+1, j, k+1] - pi[i+1, j, k-1])) / h
                pi_xy_p = 0.5 * ((pi[i+1, j+1, k] - pi[i, j+1, k]) - (pi[i+1, j, k] - pi[i, j, k])) / h**2
                pi_xz_p = 0.5 * ((pi[i+1, j, k+1] - pi[i, j, k+1]) - (pi[i+1, j, k] - pi[i, j, k])) / h**2
                pi_yy_p = 0.5 * (
                    (pi[i, j+1, k] - 2*pi[i, j, k] + pi[i, j-1, k]) +
                    (pi[i+1, j+1, k] - 2*pi[i+1, j, k] + pi[i+1, j-1, k])
                ) / h**2
                pi_zz_p = 0.5 * (
                    (pi[i, j, k+1] - 2*pi[i, j, k] + pi[i, j, k-1]) +
                    (pi[i+1, j, k+1] - 2*pi[i+1, j, k] + pi[i+1, j, k-1])
                ) / h**2
                Fx_plus = p * pi_x_p - r * (pi_xy_p * pi_y_p + pi_xz_p * pi_z_p) + r * pi_x_p * (pi_yy_p + pi_zz_p)

                # ===== Fx at i-1/2 =====
                pi_x_m = (pi[i, j, k] - pi[i-1, j, k]) / h
                pi_y_m = 0.25 * ((pi[i, j+1, k] - pi[i, j-1, k]) + (pi[i-1, j+1, k] - pi[i-1, j-1, k])) / h
                pi_z_m = 0.25 * ((pi[i, j, k+1] - pi[i, j, k-1]) + (pi[i-1, j, k+1] - pi[i-1, j, k-1])) / h
                pi_xy_m = 0.5 * ((pi[i, j+1, k] - pi[i-1, j+1, k]) - (pi[i, j, k] - pi[i-1, j, k])) / h**2
                pi_xz_m = 0.5 * ((pi[i, j, k+1] - pi[i-1, j, k+1]) - (pi[i, j, k] - pi[i-1, j, k])) / h**2
                pi_yy_m = 0.5 * (
                    (pi[i, j+1, k] - 2*pi[i, j, k] + pi[i, j-1, k]) +
                    (pi[i-1, j+1, k] - 2*pi[i-1, j, k] + pi[i-1, j-1, k])
                ) / h**2
                pi_zz_m = 0.5 * (
                    (pi[i, j, k+1] - 2*pi[i, j, k] + pi[i, j, k-1]) +
                    (pi[i-1, j, k+1] - 2*pi[i-1, j, k] + pi[i-1, j, k-1])
                ) / h**2
                Fx_minus = p * pi_x_m - r * (pi_xy_m * pi_y_m + pi_xz_m * pi_z_m) + r * pi_x_m * (pi_yy_m + pi_zz_m)

                # ===== Fy at j+1/2 =====
                pi_y_p = (pi[i, j+1, k] - pi[i, j, k]) / h
                pi_x_p = 0.25 * ((pi[i+1, j, k] - pi[i-1, j, k]) + (pi[i+1, j+1, k] - pi[i-1, j+1, k])) / h
                pi_z_p = 0.25 * ((pi[i, j, k+1] - pi[i, j, k-1]) + (pi[i, j+1, k+1] - pi[i, j+1, k-1])) / h
                pi_xy_p = 0.5 * ((pi[i+1, j+1, k] - pi[i+1, j, k]) - (pi[i, j+1, k] - pi[i, j, k])) / h**2
                pi_yz_p = 0.5 * ((pi[i, j+1, k+1] - pi[i, j, k+1]) - (pi[i, j+1, k] - pi[i, j, k])) / h**2
                pi_xx_p = 0.5 * (
                    (pi[i+1, j, k] - 2*pi[i, j, k] + pi[i-1, j, k]) +
                    (pi[i+1, j+1, k] - 2*pi[i, j+1, k] + pi[i-1, j+1, k])
                ) / h**2
                pi_zz_p = 0.5 * (
                    (pi[i, j, k+1] - 2*pi[i, j, k] + pi[i, j, k-1]) +
                    (pi[i, j+1, k+1] - 2*pi[i, j+1, k] + pi[i, j+1, k-1])
                ) / h**2
                Fy_plus = p * pi_y_p - r * (pi_xy_p * pi_x_p + pi_yz_p * pi_z_p) + r * pi_y_p * (pi_xx_p + pi_zz_p)

                # ===== Fy at j-1/2 =====
                pi_y_m = (pi[i, j, k] - pi[i, j-1, k]) / h
                pi_x_m = 0.25 * ((pi[i+1, j, k] - pi[i-1, j, k]) + (pi[i+1, j-1, k] - pi[i-1, j-1, k])) / h
                pi_z_m = 0.25 * ((pi[i, j, k+1] - pi[i, j, k-1]) + (pi[i, j-1, k+1] - pi[i, j-1, k-1])) / h
                pi_xy_m = 0.5 * ((pi[i+1, j, k] - pi[i+1, j-1, k]) - (pi[i, j, k] - pi[i, j-1, k])) / h**2
                pi_yz_m = 0.5 * ((pi[i, j, k+1] - pi[i, j-1, k+1]) - (pi[i, j, k] - pi[i, j-1, k])) / h**2
                pi_xx_m = 0.5 * (
                    (pi[i+1, j, k] - 2*pi[i, j, k] + pi[i-1, j, k]) +
                    (pi[i+1, j-1, k] - 2*pi[i, j-1, k] + pi[i-1, j-1, k])
                ) / h**2
                pi_zz_m = 0.5 * (
                    (pi[i, j, k+1] - 2*pi[i, j, k] + pi[i, j, k-1]) +
                    (pi[i, j-1, k+1] - 2*pi[i, j-1, k] + pi[i, j-1, k-1])
                ) / h**2
                Fy_minus = p * pi_y_m - r * (pi_xy_m * pi_x_m + pi_yz_m * pi_z_m) + r * pi_y_m * (pi_xx_m + pi_zz_m)

                # ===== Fz at k+1/2 =====
                pi_z_p = (pi[i, j, k+1] - pi[i, j, k]) / h
                pi_x_p = 0.25 * ((pi[i+1, j, k] - pi[i-1, j, k]) + (pi[i+1, j, k+1] - pi[i-1, j, k+1])) / h
                pi_y_p = 0.25 * ((pi[i, j+1, k] - pi[i, j-1, k]) + (pi[i, j+1, k+1] - pi[i, j-1, k+1])) / h
                pi_xz_p = 0.5 * ((pi[i+1, j, k+1] - pi[i+1, j, k]) - (pi[i, j, k+1] - pi[i, j, k])) / h**2
                pi_yz_p = 0.5 * ((pi[i, j+1, k+1] - pi[i, j+1, k]) - (pi[i, j, k+1] - pi[i, j, k])) / h**2
                pi_xx_p = 0.5 * (
                    (pi[i+1, j, k] - 2*pi[i, j, k] + pi[i-1, j, k]) +
                    (pi[i+1, j, k+1] - 2*pi[i, j, k+1] + pi[i-1, j, k+1])
                ) / h**2
                pi_yy_p = 0.5 * (
                    (pi[i, j+1, k] - 2*pi[i, j, k] + pi[i, j-1, k]) +
                    (pi[i, j+1, k+1] - 2*pi[i, j, k+1] + pi[i, j-1, k+1])
                ) / h**2
                Fz_plus = p * pi_z_p - r * (pi_xz_p * pi_x_p + pi_yz_p * pi_y_p) + r * pi_z_p * (pi_xx_p + pi_yy_p)

                # ===== Fz at k-1/2 =====
                pi_z_m = (pi[i, j, k] - pi[i, j, k-1]) / h
                pi_x_m = 0.25 * ((pi[i+1, j, k] - pi[i-1, j, k]) + (pi[i+1, j, k-1] - pi[i-1, j, k-1])) / h
                pi_y_m = 0.25 * ((pi[i, j+1, k] - pi[i, j-1, k]) + (pi[i, j+1, k-1] - pi[i, j-1, k-1])) / h
                pi_xz_m = 0.5 * ((pi[i+1, j, k] - pi[i+1, j, k-1]) - (pi[i, j, k] - pi[i, j, k-1])) / h**2
                pi_yz_m = 0.5 * ((pi[i, j+1, k] - pi[i, j+1, k-1]) - (pi[i, j, k] - pi[i, j, k-1])) / h**2
                pi_xx_m = 0.5 * (
                    (pi[i+1, j, k-1] - 2*pi[i, j, k-1] + pi[i-1, j, k-1]) +
                    (pi[i+1, j, k] - 2*pi[i, j, k] + pi[i-1, j, k])
                ) / h**2
                pi_yy_m = 0.5 * (
                    (pi[i, j+1, k-1] - 2*pi[i, j, k-1] + pi[i, j-1, k-1]) +
                    (pi[i, j+1, k] - 2*pi[i, j, k] + pi[i, j-1, k])
                ) / h**2
                Fz_minus = p * pi_z_m - r * (pi_xz_m * pi_x_m + pi_yz_m * pi_y_m) + r * pi_z_m * (pi_xx_m + pi_yy_m)


                # ===== Final divergence =====
                divF[i, j, k] = (Fx_plus - Fx_minus + Fy_plus - Fy_minus + Fz_plus - Fz_minus) / h + q*b[i,j,k]

    return divF

from numba import njit, prange, float32
import numpy as np
import numpy.typing as npt

@njit(
    ["f4(f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def Deff(
    pi: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32, 
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
) -> np.float32:
    
    p = np.float32(alphaB * (2. * alphaM - alphaB) - C2)
    r = np.float32(0.25 * C4 / (a * a * H)**2)
    h = np.float32(1./pi.shape[0])

    lchi = laplacian.operator(pi,h)
    f1 = np.float32(2.*r/3.)
    utils.linear_operator_inplace(lchi,f1,p)
    lchi = lchi.astype(np.float32)
    return np.max(lchi)

    
    

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
    ) -> npt.NDArray[np.float32]:

    """Residual of Quadratic operator

    R = -(a pi^2 + b pi + c) \\
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
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    M : np.float32
        time-dependent Planck mass
        
    Returns
    -------
    npt.NDArray[np.float32]
        Residual(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    """

    return -1*operator(pi,b,h,C2,C4,alphaB,alphaM,H,a)
    

@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_error(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
    ) -> np.float32:

    """Error on half of the residual of the quadratic operator  \\
    residual = -(a pi^2 + b pi + c)  \\
    error = sqrt[sum(residual**2)] \\
    
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
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    M : np.float32
        time-dependent Planck mass
    Returns
    -------
    np.float32
        Residual error

    """
    ncells_1d = pi.shape[0]
    res = operator(pi,b,h,C2,C4,alphaB,alphaM,H,a)
    result = 0.0

    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):
                result += res[i,j,k]**2
    
    return np.sqrt(result)
    

 
def smoothing(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    n_smoothing: int) -> None:
    
    """Smooth Chi field with several Jacobi iterations

    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    n_smoothing : int
        number of smoothing iterations

    """

    frac = 0.05
    h = np.float32(1./pi.shape[0])
    
    
    for nn in range(n_smoothing):
        #pi = pi + delt*operator(pi,b,h,C2,C4,alphaB,alphaM,H,a)
        delt = Deff(pi,h,C2,C4,alphaB,alphaM,H,a)
        #print(nn,delt)
        delt = np.float32(1*frac*h*h/delt)
        #print(nn,delt)
        utils.linear_operator_vectors_inplace(pi,np.float32(1.),operator(pi,b,h,C2,C4,alphaB,alphaM,H,a),delt)
        
    

def smoothing_with_rhs(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    n_smoothing: int,
    rhs: npt.NDArray[np.float32]) -> None:
    
    """Smooth Chi field with several Jacobi iterations with rhs

    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    n_smoothing : int
        number of smoothing iterations

    """

    frac = 0.05
    h = np.float32(1./pi.shape[0])
    
    
    for nn in range(n_smoothing):
        delt = Deff(pi,h,C2,C4,alphaB,alphaM,H,a)
        #print(delt,nn)
        delt = np.float32(1*frac*h*h/delt)
        #print(delt,nn)
        op = operator(pi,b,h,C2,C4,alphaB,alphaM,H,a)
        #print(op[0,0,0])
        utils.add_vector_scalar_inplace(op,rhs,np.float32(-1.))
        utils.linear_operator_vectors_inplace(pi,np.float32(1.),op,delt)
    
    

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    alphaB: np.float32,
    alphaM: np.float32
) -> npt.NDArray[np.float32]:
    """
    HG: 14/11/2024

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
    

    Returns
    -------
    npt.NDArray[np.float32]
        Chi field

    
    """
    
    lfac = (alphaB - alphaM) / (2*alphaB*alphaM - alphaB**2 - C2)
    pi = np.empty_like(b)
    one_by_six = np.float32(1./6)
    ncells_1d = b.shape[0]
    for i in prange(-1,ncells_1d-1):
        for j in prange(-1,ncells_1d-1):
            for k in prange(-1,ncells_1d-1):
                pi[i, j, k] = - one_by_six*lfac*h*h*b[i,j,k]
    
    return pi


