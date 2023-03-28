import logging
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

import mesh
import utils


# @utils.profile_me
def pm(
    position: npt.NDArray[np.float32],
    param: pd.Series,
    potential: npt.NDArray[np.float32] = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compute Particle-Mesh acceleration

    Args:
        position (npt.NDArray[np.float32]): Positions [3,N_part]
        param (pd.Series): Parameter container
        potential (npt.NDArray[np.float32], optional): Gravitational potential [N_cells_1d, N_cells_1d,N_cells_1d]
        Defaults to None.

    Returns:
        tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]: acceleration, potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    logging.debug("In pm")
    # Interpolation order
    if param["interp"].casefold() == "NGP".casefold():
        func_interp = mesh.NGP
        func_inv_interp = mesh.invNGP_vec
    elif param["interp"].casefold() == "CIC".casefold():
        func_interp = mesh.CIC
        func_inv_interp = mesh.invCIC_vec
    elif param["interp"].casefold() == "TSC".casefold():
        func_interp = mesh.TSC
        func_inv_interp = mesh.invTSC_vec
    else:
        raise ValueError("ERROR: Interpolation must be 'NGP', 'CIC' or 'TSC'")

    ncells_1d = 2 ** (param["ncoarse"])
    h = np.float32(1.0 / ncells_1d)
    # Compute density mesh from particles
    f1 = 1.5 * param["aexp"] * param["Om_m"]
    f2 = param["mpart"] * ncells_1d**3 / (param["unit_l"] ** 3 * param["unit_d"])
    rhs = func_interp(position, ncells_1d)
    utils.density_renormalize(rhs, f1, f2)

    # print("COmpute residual....")
    res1 = mesh.residual((-1.0 / 6 * h**2) * rhs, rhs, h).ravel()
    # print(f"Residual, density approx {np.dot(res, res)}")
    if potential is not None:
        res2 = mesh.residual(potential, rhs, h).ravel()
        # print(f"Residual, previous {np.dot(res, res)}")
        res3 = mesh.residual(
            (param["aexp"] / param["aexp_old"]) * potential, rhs, h
        ).ravel()
        print(
            f"{param['aexp']} {np.dot(res1, res1)} {np.dot(res2, res2)} {np.dot(res3, res3)} # residuals"
        )
    # Initialise Potential if there is no previous step
    if potential is None:
        potential = utils.prod_vector_scalar(rhs, (-1.0 / 6 * h**2))

    # Main procedure: Poisson solver
    if param["poisson_solver"].casefold() == "mg".casefold():
        potential = multigrid(potential, rhs, h, param)
    elif param["poisson_solver"].casefold() == "fft".casefold():
        potential = fft(rhs, h, param)
    elif param["poisson_solver"].casefold() == "cg".casefold():
        potential = conjugate_gradient(potential, rhs, h, param)
    elif param["poisson_solver"].casefold() == "sd".casefold():
        potential = steepest_descent(potential, rhs, h, param)
    else:
        raise ValueError(
            "ERROR: Only 'mg', 'fft', 'cg' or 'sd' solvers for the moment..."
        )

    # Compute Force
    force = mesh.derivative(potential)
    acceleration = func_inv_interp(force, position)  # In BU, particle mass = 1
    return (acceleration, potential)  # return acceleration


@utils.time_me
def multigrid(
    x: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute Multigrid

    Args:
        x (npt.NDArray[np.float32]): Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
        rhs (npt.NDArray[np.float32]): Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
        h (float): grid size
        param (pd.Series): Parameter container

    Returns:
        npt.NDArray[np.float32]: Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    # TODO:  - Check w_relax
    #        - Inplace instead of returning function
    #        - Parallelize (test PyOMP)
    #        - Output Energy ! Check conservation
    #        - Define types in decorator
    #        - Check if we get better initial residual with scaled potential from previous step
    # Compute tolerance
    # if param["tolerance"] == 0:
    if not "tolerance" in param:
        trunc = mesh.truncation(rhs, h).ravel()
        param["tolerance"] = param["epsrel"] * np.sqrt(np.dot(trunc, trunc))

    # Main procedure: Multigrid
    if param["cycle"].casefold() == "V".casefold():
        func_cycle = mesh.V_cycle
    elif param["cycle"].casefold() == "F".casefold():
        func_cycle = mesh.F_cycle
    elif param["cycle"].casefold() == "W".casefold():
        func_cycle = mesh.W_cycle
    else:
        raise ValueError("ERROR: Cycle must be 'V', 'F' or 'W'")
    # Cycling
    for _ in range(param["n_cycle"]):
        potential = func_cycle(
            x,
            rhs,
            0,
            param,
        )
        # residual = mesh.residual(potential, rhs, h).ravel()
        # residual_error = np.sqrt(np.dot(residual, residual))
        residual_error = mesh.residual_error_half(potential, rhs, h)
        print(f"{residual_error=}")
        if residual_error < param["tolerance"]:
            break
    return potential


def fft(
    rhs: npt.NDArray[np.float32], h: np.float32, param: pd.Series
) -> npt.NDArray[np.float32]:
    """Compute FFT

    Args:
        rhs (npt.NDArray[np.float32]): Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
        h (np.float32): grid size
        param (pd.Series): Parameter container

    Returns:
        npt.NDArray[np.float32]: Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    # TODO: Rewrite with pyFFTw
    logging.debug("In fft")
    # TO DO: - Only compute invk2 once...
    # FFT
    rhs_fourier = np.fft.rfftn(rhs)
    # Divide by k**2
    n = 2 ** param["ncoarse"]
    k0 = 2.0 * np.pi * np.fft.rfftfreq(n, 1 / n)
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, 1 / n)
    k2 = (
        k0[np.newaxis, np.newaxis, :] ** 2
        + k1[:, np.newaxis, np.newaxis] ** 2
        + k1[np.newaxis, :, np.newaxis] ** 2
    )
    invk2 = np.where(k2 == 0, 0, k2 ** (-1))
    potential_fourier = -rhs_fourier * invk2
    # Inverse FFT
    potential = np.fft.irfftn(potential_fourier).astype(np.float32)

    return potential


def conjugate_gradient(
    x: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute Conjugate Gradient

    Args:
        x (npt.NDArray[np.float32]): Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
        rhs (npt.NDArray[np.float32]): Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
        h (float): grid size
        param (pd.Series): Parameter container

    Returns:
        npt.NDArray[np.float32]: Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    logging.debug("In conjugate_gradient")
    Nmax = 1000
    ncells_1d = int(x.shape[0])
    # Run
    potential = x.ravel()
    r = mesh.residual(x, rhs, h).ravel()
    d = r
    rrold = np.dot(r, r)

    if param["tolerance"] == 0:
        param["tolerance"] = param["epsrel"] * np.sqrt(rrold)

    for i in range(Nmax):
        Ad = mesh.laplacian(d.reshape(ncells_1d, ncells_1d, ncells_1d), h).ravel()
        alpha = rrold / np.dot(d, Ad)
        potential = potential + np.dot(alpha, d)
        # plt.imshow(potential.reshape(ncells_1d, ncells_1d, ncells_1d)[0])
        # plt.show()
        if (i != 0) and ((i % 50) == 0):
            r = mesh.residual(
                potential.reshape(ncells_1d, ncells_1d, ncells_1d), rhs, h
            ).ravel()
        else:
            r = r - np.dot(alpha, Ad)

        rrnew = np.dot(r, r)
        logging.debug(
            f"{i=} {np.sqrt(rrnew)=} {param['tolerance']=} Still within tolerance"
        )
        if np.sqrt(rrnew) < param["tolerance"]:
            logging.debug(f"{i} {np.sqrt(rrnew)=} {param['tolerance']}")
            break
        d = r + rrnew / rrold * d
        rrold = rrnew
    return potential.reshape(ncells_1d, ncells_1d, ncells_1d)


def steepest_descent(
    x: npt.NDArray[np.float32],
    rhs: npt.NDArray[np.float32],
    h: np.float32,
    param: pd.Series,
) -> npt.NDArray[np.float32]:
    """Compute Steepest descent

    Args:
        x (npt.NDArray[np.float32]): Potential (first guess) [N_cells_1d, N_cells_1d,N_cells_1d]
        rhs (npt.NDArray[np.float32]): Right-hand side of Poisson Equation (density) [N_cells_1d, N_cells_1d,N_cells_1d]
        h (float): grid size
        param (pd.Series): Parameter container

    Returns:
        npt.NDArray[np.float32]: Potential [N_cells_1d, N_cells_1d,N_cells_1d]
    """
    logging.debug("In steepest_descent")
    Nmax = 10000
    ncells_1d = int(x.shape[0])
    # Run
    potential = x.ravel()
    r = mesh.residual(x, rhs, h).ravel()
    rr = np.dot(np.transpose(r), r)

    if param["tolerance"] == 0:
        param["tolerance"] = param["epsrel"] * np.sqrt(rr)

    for i in range(Nmax):
        Ar = mesh.laplacian(r.reshape(ncells_1d, ncells_1d, ncells_1d), h).ravel()
        alpha = rr / np.dot(np.transpose(r), Ar)
        potential = potential + np.dot(alpha, r)
        # plt.imshow(potential.reshape(ncells_1d, ncells_1d, ncells_1d)[0])
        # plt.show()
        if (i != 0) and ((i % 50) == 0):
            r = mesh.residual(
                potential.reshape(ncells_1d, ncells_1d, ncells_1d), rhs, h
            ).ravel()
        else:
            r = r - np.dot(alpha, Ar)

        rr = np.dot(np.transpose(r), r)
        logging.debug(
            f"{i=} {np.sqrt(rr)=} {param['tolerance']=} Still within tolerance"
        )
        if np.sqrt(rr) < param["tolerance"]:
            logging.debug(f"{i} {np.sqrt(rr)=} {param['tolerance']}")
            break
    return potential.reshape(ncells_1d, ncells_1d, ncells_1d)
