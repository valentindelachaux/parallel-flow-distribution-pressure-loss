import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import torch
import fds_torch as fdst

def Kxin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff): #Din, Dx, theta in par
    # return torch tensor unit
    return torch.tensor(1.0, requires_grad=True)
    # return coeff*fdst.K_branch_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)

def Kyin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"):
    return torch.tensor(1.0, requires_grad=True)
    # return coeff*fdst.K_run_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)

def Kxout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff):
    return torch.tensor(1.0, requires_grad=True)
    # return coeff*fdst.K_branch_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)

def Kyout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff):
    return torch.tensor(1.0, requires_grad=True)
    # return coeff*fdst.K_run_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)

def calc(q_vect, par, cond, series=None, simplified=False):
    # Check if q_vect is a tensor, if not convert it
    if not isinstance(q_vect, torch.Tensor):
        q_vect = torch.tensor(q_vect, dtype=torch.float32, requires_grad=True)

    # Extracting parameters and ensuring they are tensors
    N = par["N"]
    ref = par["ref"]
    theta = par["theta"]
    Ax = par["A_riser"]
    Ain = par["A_man"]
    Aout = par["A_man"]
    Dx = par["D_riser"]
    Din = par["D_man"]
    Dout = par["D_man"]
    c_Kxin = par["coeff_Kxin"]
    c_Kxout = par["coeff_Kxout"]
    c_Kyin = par["coeff_Kyin"]
    c_Kyout = par["coeff_Kyout"]
    eta = cond["eta"]
    ep = par["roughness"]
    rho = cond["rho"]

    DR = par["DR"]

    # Series calculations
    if series is None:
        QF = cond["Dv"]
        QF_out = 0
        alpha = 1
    else:
        QF, QF_out, alpha = series

    # Convert series values to tensors if they are not already
    QF = torch.tensor(QF, dtype=torch.float32, requires_grad=True)
    QF_out = torch.tensor(QF_out, dtype=torch.float32, requires_grad=True)
    alpha = torch.tensor(alpha, dtype=torch.float32, requires_grad=True)

    # Calculations
    Qin = QF - torch.tensor([torch.sum(q_vect[i+1:]) for i in range(N)], dtype=torch.float32, requires_grad=True)

    
    if ref == 0:  # Z-type
        Qout = QF_out + torch.tensor([torch.sum(q_vect[i:N]) for i in range(N)], dtype=torch.float32)
    else:  # U-type
        Qout = QF_out + torch.tensor([torch.sum(q_vect[:i+1]) for i in range(N)], dtype=torch.float32)

    uin = Qin / Ain
    ux = q_vect / Ax
    uout = Qout / Aout

    # Assuming fds.core.Reynolds is modified for PyTorch
    Rein = fdst.Reynolds(uin, Din, rho, mu=eta)
    Rex = fdst.Reynolds(ux, Dx, rho, mu=eta)
    Reout = fdst.Reynolds(uout, Dout, rho, mu=eta)

    Lex = 0.05 * torch.sqrt(4 * torch.tensor(Ax, dtype=torch.float32) / torch.pi) * Rex

    # Assuming fds.friction.friction_factor is modified for PyTorch
    fin = torch.tensor([fdst.friction_factor(Re=Rein[i], eD=ep/Din) for i in range(N)], dtype=torch.float32, requires_grad=True)
    fx = torch.tensor([fdst.friction_factor(Re=Rex[i], eD=ep/Dx) for i in range(N)], dtype=torch.float32, requires_grad=True)
    fout = torch.tensor([fdst.friction_factor(Re=Reout[i], eD=ep/Dout) for i in range(N)], dtype=torch.float32, requires_grad=True)

    Kx_in = torch.tensor([Kxin(Din, Dx, theta, Qin[i], q_vect[i], i, c_Kxin) for i in range(N)], dtype=torch.float32, requires_grad=True)

    if series is None:
        Ky_in = torch.tensor([0]+[Kyin(Din,Dx,theta,Qin[i-1],q_vect[i],i,c_Kyin) for i in range(1,N)],dtype=torch.float32, requires_grad=True)
    else :
        Ky_in = torch.tensor([Kyin(Din,Dx,theta,QF*(1-alpha),q_vect[0],0,c_Kyin)]+[Kyin(Din,Dx,theta,Qin[i-1],q_vect[i],i,c_Kyin) for i in range(1,N)],dtype=torch.float32, requires_grad=True)

    Kx_out = torch.tensor([Kxout(Dout, Dx, theta, Qout[i], q_vect[i], i, c_Kxout) for i in range(N)], dtype=torch.float32, requires_grad=True)
    Ky_out = torch.tensor([Kyout(Dout, Dx, theta, Qout[i], q_vect[i], i, c_Kyout) for i in range(N)], dtype=torch.float32, requires_grad=True)

    Kse = fdst.contraction_sharp(DR * Din, Din, fd=fin[N - 1], Re=Rein[N - 1], roughness=ep)
    K_se = Kse * torch.tensor([0.05, 0.1, 0.25, 0.6], dtype=torch.float32, requires_grad=True)
    K_se = torch.cat((torch.zeros(N - len(K_se)), K_se))

    var_dict = {'Qin': Qin, 'Qout': Qout, 'uin': uin, 'ux': ux, 'uout': uout, 'Rein': Rein, 'Rex': Rex, 'Reout': Reout, 'fin': fin, 'fx': fx, 'fout': fout, 'Kx_in': Kx_in, 'Ky_in': Ky_in, 'Kx_out': Kx_out, 'Ky_out': Ky_out, 'K_se': K_se, 'Lex': Lex}

    return var_dict

def fun(x, par, cond, series=None):
    """
    Convert a system of equations to a PyTorch-compatible loss function.

    Args:
        x: torch.Tensor, system state in the format [Pin_0, ..., Pin_N-1, Pout_0, ..., Pout_N-1, q_0, ..., q_N-1]
        simplified: bool, flag to determine whether to use a simplified model or not
        par: dict, parameters of the system
        cond: dict, conditions applied to the system
        DR: float, diameter ratio or similar parameter
        series: bool or similar parameter indicating series configuration
        rho: float, fluid density
        N: int, number of elements in the system
        Ly_list: list, list of lengths
        Din: float, diameter of the inlet
        Dout: float, diameter of the outlet
        QF: float, total flow rate
        alpha: float, scaling factor for the flow rate
        ref: int, reference point index

    Returns:
        loss: torch.Tensor, the computed loss based on the system equations
    """

    # Parameters
    N = par["N"]
    if series is None:
        QF = cond["Dv"]
        QF_out = 0
        alpha = 1
    else :
        QF = series[0]
        QF_out = series[1]
        alpha = series[2]

    ref = par["ref"] # 0 (en Z) ou N-1 (en U)

    Dx = par["D_riser"]
    Din = par["D_man"]
    Dout = par["D_man"]
    Ain = par["A_man"]
    Aout = par["A_man"]
    Lx = par["L_riser"]

    Ly_list = par["Ly"] # N-1 values
    rho = cond["rho"]

    DR = par['DR']
    fappx = par['fappx']

    # Initialize loss tensor
    loss = torch.tensor(0., requires_grad=True)

    # Function logic using PyTorch operations

    # Assuming calc returns tensors compatible with PyTorch
    var_dict = calc(x[2*N:3*N], par, cond)
    Qin = var_dict['Qin']; Qout = var_dict['Qout']; uin = var_dict['uin']; ux = var_dict['ux']; uout = var_dict['uout']; Rein = var_dict['Rein']; Rex = var_dict['Rex']; Reout = var_dict['Reout']; fin = var_dict['fin']; fx = var_dict['fx']; fout = var_dict['fout']; Kx_in = var_dict['Kx_in']; Ky_in = var_dict['Ky_in']; Kx_out = var_dict['Kx_out']; Ky_out = var_dict['Ky_out']; K_se = var_dict['K_se']; Lex = var_dict['Lex']

    for i in range(N):
        if i >= 1:
            # Pressure difference in inlet
            loss = loss + torch.square(torch.abs(x[i]) - torch.abs(x[i-1]) - (rho/2)*(uin[i-1]**2-uin[i]**2 + (fin[i]*Ly_list[i-1]*uin[i]**2)/Din + Ky_in[i]*uin[i-1]**2 + K_se[i] * uin[N-1]**2))
            
            # Pressure difference in outlet
            if ref == 0:
                loss = loss + torch.square(torch.abs(x[N+i]) - torch.abs(x[N+i-1]) - (rho/2)*(uout[i-1]**2-uout[i]**2 + (fout[i]*Ly_list[i-1]*uout[i]**2)/Dout + Ky_out[i-1]*uout[i-1]**2))
            else:
                loss = loss + torch.square(torch.abs(x[N+i-1]) - torch.abs(x[N+i]) - (rho/2)*(uout[i]**2-uout[i-1]**2 + (fout[i]*Ly_list[i-1]*uout[i-1]**2)/Dout + Ky_out[i]*uout[i]**2))

            # Pressure difference between in and out
            if par["sch"] == "exchanger":
                b_x = 0.  # Linear part
                a_x = (fx[i]*(Lx-Lex[i])+ 4*fappx*fx[i]*Lex[i])/Dx  # Second order part
            elif par["sch"] == "system":
                a_x = par["a_x"]
                b_x = par["b_x"]
            
            loss = loss + torch.square(torch.abs(x[i]) - torch.abs(x[N+i]) - (rho/2)*(uout[i]**2-uin[i]**2 + b_x*ux[i]+a_x*ux[i]**2+Kx_in[i]*uin[i]**2+Kx_out[i]*uout[i]**2))

    # Sum of flows constraint
    loss = loss + torch.square(torch.sum(torch.abs(x[2*N:3*N])) - QF*alpha)

    # Reference pressure condition
    loss = loss + torch.square(torch.abs(x[N+ref]))

    return loss

def fun_wo_square(x, par, cond, series=None):
    """
    Convert a system of equations to a PyTorch-compatible loss function.

    Args:
        x: torch.Tensor, system state in the format [Pin_0, ..., Pin_N-1, Pout_0, ..., Pout_N-1, q_0, ..., q_N-1]
        simplified: bool, flag to determine whether to use a simplified model or not
        par: dict, parameters of the system
        cond: dict, conditions applied to the system
        DR: float, diameter ratio or similar parameter
        series: bool or similar parameter indicating series configuration
        rho: float, fluid density
        N: int, number of elements in the system
        Ly_list: list, list of lengths
        Din: float, diameter of the inlet
        Dout: float, diameter of the outlet
        QF: float, total flow rate
        alpha: float, scaling factor for the flow rate
        ref: int, reference point index

    Returns:
        loss: torch.Tensor, the computed loss based on the system equations
    """

    # Parameters
    N = par["N"]
    if series is None:
        QF = cond["Dv"]
        QF_out = 0
        alpha = 1
    else :
        QF = series[0]
        QF_out = series[1]
        alpha = series[2]

    ref = par["ref"] # 0 (en Z) ou N-1 (en U)

    Dx = par["D_riser"]
    Din = par["D_man"]
    Dout = par["D_man"]
    Ain = par["A_man"]
    Aout = par["A_man"]
    Lx = par["L_riser"]

    Ly_list = par["Ly"] # N-1 values
    rho = cond["rho"]

    DR = par['DR']
    fappx = par['fappx']

    # Initialize loss tensor
    loss = torch.tensor(0., requires_grad=True)

    # Function logic using PyTorch operations

    # Assuming calc returns tensors compatible with PyTorch
    var_dict = calc(x[2*N:3*N], par, cond)
    Qin = var_dict['Qin']; Qout = var_dict['Qout']; uin = var_dict['uin']; ux = var_dict['ux']; uout = var_dict['uout']; Rein = var_dict['Rein']; Rex = var_dict['Rex']; Reout = var_dict['Reout']; fin = var_dict['fin']; fx = var_dict['fx']; fout = var_dict['fout']; Kx_in = var_dict['Kx_in']; Ky_in = var_dict['Ky_in']; Kx_out = var_dict['Kx_out']; Ky_out = var_dict['Ky_out']; K_se = var_dict['K_se']; Lex = var_dict['Lex']

    for i in range(N):
        if i >= 1:
            # Pressure difference in inlet
            loss = loss + torch.abs(torch.abs(x[i]) - torch.abs(x[i-1]) - (rho/2)*(uin[i-1]**2-uin[i]**2 + (fin[i]*Ly_list[i-1]*uin[i]**2)/Din + Ky_in[i]*uin[i-1]**2 + K_se[i] * uin[N-1]**2))
            
            # Pressure difference in outlet
            if ref == 0:
                loss = loss + torch.abs(torch.abs(x[N+i]) - torch.abs(x[N+i-1]) - (rho/2)*(uout[i-1]**2-uout[i]**2 + (fout[i]*Ly_list[i-1]*uout[i]**2)/Dout + Ky_out[i-1]*uout[i-1]**2))
            else:
                loss = loss + torch.abs(torch.abs(x[N+i-1]) - torch.abs(x[N+i]) - (rho/2)*(uout[i]**2-uout[i-1]**2 + (fout[i]*Ly_list[i-1]*uout[i-1]**2)/Dout + Ky_out[i]*uout[i]**2))

            # Pressure difference between in and out
            if par["sch"] == "exchanger":
                b_x = 0.  # Linear part
                a_x = (fx[i]*(Lx-Lex[i])+ 4*fappx*fx[i]*Lex[i])/Dx  # Second order part
            elif par["sch"] == "system":
                a_x = par["a_x"]
                b_x = par["b_x"]
            
            loss = loss + torch.abs(torch.abs(x[i]) - torch.abs(x[N+i]) - (rho/2)*(uout[i]**2-uin[i]**2 + b_x*ux[i]+a_x*ux[i]**2+Kx_in[i]*uin[i]**2+Kx_out[i]*uout[i]**2))

    # Sum of flows constraint
    loss = loss + torch.abs(torch.sum(torch.abs(x[2*N:3*N])) - QF*alpha)

    # Reference pressure condition
    loss = loss + torch.abs(torch.abs(x[N+ref]))

    return loss

def fun_simplified(x, par, cond, series=None):

    # Parameters
    N = par["N"]
    if series is None:
        QF = cond["Dv"]
        QF_out = 0
        alpha = 1
    else :
        QF = series[0]
        QF_out = series[1]
        alpha = series[2]

    ref = par["ref"] # 0 (en Z) ou N-1 (en U)

    Dx = par["D_riser"]
    Din = par["D_man"]
    Dout = par["D_man"]
    Ain = par["A_man"]
    Aout = par["A_man"]
    Lx = par["L_riser"]

    Ly_list = par["Ly"] # N-1 values
    rho = cond["rho"]

    DR = par['DR']
    fappx = par['fappx']

    # Initialize loss tensor
    loss = torch.tensor(0., requires_grad=True)

    loss = loss + torch.square(torch.sum(torch.abs(x[2*N:3*N])) - QF*alpha)

    loss = loss + torch.abs(torch.abs(x[N+ref]))

    return loss


def init_qx(par, cond, q_init):

    # Parameters
    eps = cond["eps"]

    N = par["N"]
    QF = cond["Dv"]

    ref = par["ref"]
    theta = float(par["theta"])

    Ax = par["A_riser"]
    Ain = par["A_man"]
    Aout = par["A_man"]
    Dx = par["D_riser"]
    Din = par["D_man"]
    Dout = par["D_man"]
    Lx = par["L_riser"]

    c_Kxin = par["coeff_Kxin"]
    c_Kxout = par["coeff_Kxout"]
    c_Kyin = par["coeff_Kyin"]
    c_Kyout = par["coeff_Kyout"]

    Ly_list = par["Ly"] # N-1 values

    eta = cond["eta"] # dynamic viscosity

    ep = par["roughness"]

    rho = cond["rho"]

    # Start

    N = par["N"]
    qx = np.zeros(N)
    P_in = np.zeros(N)
    P_out = np.zeros(N)

    # Initialisation avec des débits
    if q_init == []:
        for i in range(N):
            qx[i] = QF/N
    else:
        for i in range(N):
            qx[i] = q_init[i]
    
    i = ref # ref = 0 or N-1
    Qin_ref = sum([qx[j] for j in range(0,i+1)])
    Qout_ref = sum([qx[j] for j in range(i,N)])

    uin_ref = Qin_ref/Ain
    ux_ref = qx[i]/Ax
    uout_ref = Qout_ref/Aout

    Rein_ref = fds.core.Reynolds(uin_ref,Din,rho,mu=eta)
    Rex_ref = fds.core.Reynolds(ux_ref,Dx,rho,mu=eta)
    Reout_ref = fds.core.Reynolds(uout_ref,Dout,rho,mu=eta)
    fin_ref = fds.friction.friction_factor(Re = Rein_ref)
    fx_ref = fds.friction.friction_factor(Re = Rex_ref)
    fout_ref = fds.friction.friction_factor(Re = Reout_ref)

    Kxin_ref = Kxin(Din,Dx,theta,Qin_ref,qx[i],ref,c_Kxin)
    Kxout_ref = Kxout(Dout,Dx,theta,Qout_ref,qx[i],ref,c_Kxout)

    P_out[ref] = 0
    DPx_ref = (rho/2)*(fx_ref*(Lx/Dx)*ux_ref**2+Kxin_ref*uin_ref**2+Kxout_ref*uout_ref**2)
    P_in[ref] = DPx_ref+P_out[ref]

    if ref == 0:
        ra = range(1,N)
    else:
        ra = range(N-2,-1,-1)

    for i in ra:
        Qin_i = sum([qx[j] for j in range(0,i)])
        Qout_i = sum([qx[j] for j in range(i,N)])
        
        uin_i = Qin_i/Ain
        ux_i = qx[i]/Ax
        uout_i = Qout_i/Aout

        Rein_i = fds.core.Reynolds(uin_i,Din,rho,mu=eta)
        Rex_i = fds.core.Reynolds(ux_i,Dx,rho,mu=eta)
        Reout_i = fds.core.Reynolds(uout_i,Dout,rho,mu=eta)
        fin_i = fds.friction.friction_factor(Re = Rein_i)
        fx_i = fds.friction.friction_factor(Re = Rex_i)
        fout_i = fds.friction.friction_factor(Re = Reout_i)

        Kxin_i = Kxin(Din,Dx,theta,Qin_i,qx[i],i,c_Kxin)
        if i >= 1: # useless because the loop is for i = 1 to N-1 included
            Qin_im = sum([qx[j] for j in range(0,i+1)])
            Qout_im = sum([qx[j] for j in range(i,N)])
            uin_im = Qin_im/Ain
            uout_im = Qout_im/Aout
            Kyin_i = Kyin(Din,Dx,theta,Qout_im,qx[i],i,c_Kyin)
        Kxout_i = Kxout(Dout,Dx,theta,Qout_i,qx[i],i,c_Kxout)
        Kyout_i = Kyout(Dout,Dx,theta,Qout_i,qx[i],i,c_Kyout)

        if par["sch"] == "exchanger":
            b_x = 0. # linear part
            a_x = fx_i*(Lx/Dx) # second order part
        elif par["sch"] == "system":
            a_x = par["a_x"]
            b_x = par["b_x"]

        P_in[i] = P_in[i-1] + (rho/2)*(uin_im**2-uin_i**2 + (fin_i*Ly_list[i-1]*uin_i**2)/Din + Kyin_i*uin_i**2)
        P_out[i] = P_out[i-1] + (rho/2)*(b_x*ux_i+a_x*ux_i**2+Kxin_i*uin_i**2+Kxout_i*uout_i**2)

    qx_tensor = torch.tensor(qx, requires_grad=True)
    P_in_tensor = torch.tensor(P_in, requires_grad=True)
    P_out_tensor = torch.tensor(P_out, requires_grad=True)

    return qx_tensor, P_in_tensor, P_out_tensor