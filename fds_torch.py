import torch
import numpy as np
from fluids.numerics import interp


def manual_torch_interp(x,y,x_target):

    # Find the indices of the two points surrounding x_target
    index = (x >= x_target).nonzero(as_tuple=True)[0][0]
    if index == 0:
        index = 1
    elif index == len(x):
        index = len(x) - 1

    # Get the two surrounding points
    x1, x2 = x[index-1], x[index]
    y1, y2 = y[index-1], y[index]

    # Perform linear interpolation
    y_target = y1 + ((x_target - x1) * (y2 - y1)) / (x2 - x1)

    return y_target

## fittings

def K_branch_diverging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90.):
    '''Returns the loss coefficient for the branch of a diverging tee or wye
    according to the Crane method.

    Parameters are described in the original function docstring.
    '''

    # Convert inputs to torch tensors if they are not already

    beta = (D_branch / D_run)
    beta2 = beta * beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch / Q_comb

    if angle < 60 or beta <= 2 / 3.:
        H, J = 1., 2.
    else:
        H, J = 0.3, 0

    if angle < 75:
        if beta2 <= 0.35:
            if Q_ratio <= 0.4:
                G = 1.1 - 0.7 * Q_ratio
            else:
                G = 0.85
        else:
            if Q_ratio <= 0.6:
                G = 1.0 - 0.6 * Q_ratio
            else:
                G = 0.6
    else:
        if beta2 <= 2 / 3.:
            G = 1
        else:
            G = 1 + 0.3 * Q_ratio * Q_ratio

    # Convert angle to radians and compute cos using torch function
    angle_rad = torch.deg2rad(torch.tensor([angle],requires_grad=True))
    K_branch = G * (1 + H * (Q_ratio / beta2) ** 2 - J * (Q_ratio / beta2) * torch.cos(angle_rad))

    # Return the K_branch tensor
    return K_branch

def K_run_diverging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90.):
    '''Returns the loss coefficient for the run of a converging tee or wye according to the Crane method.

    Parameters are described in the original function docstring.
    '''

    # Convert inputs to torch tensors if they are not already

    beta = D_branch / D_run
    beta2 = beta * beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch / Q_comb

    if beta2 <= 0.4:
        M = 0.4
    elif Q_ratio <= 0.5:
        M = 2.0 * (2.0 * Q_ratio - 1.0)
    else:
        M = 0.3 * (2.0 * Q_ratio - 1.0)
    
    K_run = M * Q_ratio * Q_ratio

    # Return the K_run
    return K_run

branch_converging_Crane_Fs = torch.tensor([1.74, 1.41, 1.0, 0.0],requires_grad=True)
branch_converging_Crane_angles = torch.tensor([30.0, 45.0, 60.0, 90.0],requires_grad=True)

def K_branch_converging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90.0):
    beta = D_branch / D_run
    beta2 = beta * beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch / Q_comb

    if beta2 <= 0.35:
        C = 1.
    elif Q_ratio <= 0.4:
        C = 0.9*(1 - Q_ratio)
    else:
        C = 0.55  

    D, E = 1.0, 2.0
    F = manual_torch_interp(branch_converging_Crane_angles, branch_converging_Crane_Fs,angle)
    K = C * (1. + D * (Q_ratio / beta2)**2 - E * (1. - Q_ratio)**2 - F / beta2 * Q_ratio**2)
    return K

run_converging_Crane_Fs = torch.tensor([1.74, 1.41, 1.0],requires_grad=True)
run_converging_Crane_angles = torch.tensor([30.0, 45.0, 60.0],requires_grad=True)

def K_run_converging_Crane(D_run, D_branch, Q_run, Q_branch, angle=90.):
    beta = D_branch / D_run
    beta2 = beta * beta
    Q_comb = Q_run + Q_branch
    Q_ratio = Q_branch / Q_comb
    
    if angle < 75.0:
        C = 1.
    else:
        return 1.55 * Q_ratio - Q_ratio * Q_ratio

    D, E = 0.0, 1.0
    F = manual_torch_interp(run_converging_Crane_angles, run_converging_Crane_Fs, angle)
    K = C * (1. + D * (Q_ratio / beta2)**2 - E * (1. - Q_ratio)**2 - F / beta2 * Q_ratio**2)
    return K

## core

def Reynolds(V, D, rho=None, mu=None, nu=None):
    if rho is not None and mu is not None:
        nu = mu / rho
    elif nu is None:
        raise ValueError('Either density and viscosity, or dynamic viscosity, \
        is needed')
    return V * D / nu

## friction factor

LAMINAR_TRANSITION_PIPE = 2040.

def friction_laminar(Re):
    return 64. / Re

def Clamond(Re, eD, fast=False):
    X1 = eD * Re * 0.1239681863354175460160858261654858382699
    X2 = torch.log(Re) - 0.7793974884556819406441139701653776731705
    F = X2 - 0.2
    X1F = X1 + F
    X1F1 = 1. + X1F

    E = (torch.log(X1F) - 0.2) / X1F1
    F = F - (X1F1 + 0.5 * E) * E * X1F / (X1F1 + E * (1. + (1.0 / 3.0) * E))

    if not fast:
        X1F = X1 + F
        X1F1 = 1. + X1F
        E = (torch.log(X1F) + F - X2) / X1F1

        b = (X1F1 + E * (1. + 1.0 / 3.0 * E))
        F = b / (b * F - ((X1F1 + 0.5 * E) * E * X1F))
        return 1.325474527619599502640416597148504422899 * (F * F)

    return 1.325474527619599502640416597148504422899 / (F * F)


def friction_factor(Re, eD=0.0):

    if Re < LAMINAR_TRANSITION_PIPE:
        f = friction_laminar(Re)
    else:
        f = Clamond(Re, eD, False)
    return f

def contraction_sharp(Di1, Di2, fd=None, Re=None, roughness=0.0, method='Rennels'):
    return torch.tensor(1.0, requires_grad=True)