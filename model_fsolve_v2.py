import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import scipy.optimize as sc

class flow_field:
    def __init__(self, X, par, cond):
        """
        Args:
            X (list): list of channel flow values
            par (dict): dictionary of heat exchanger parameters
            cond (condition): object containing the conditions of the fluid """

        self.X = np.array(X)
        self.par = par
        self.cond = cond
        N=par["N"]
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
        method = par["method"]
        ep = par["roughness"]
        alpha = cond.alpha
        eta = cond.eta
        rho = cond.rho
        Qin_c = cond.Qin_c
        Qin_d = np.sum(self.X)/alpha
        Qin = Qin_d - np.array([np.sum(self.X[i+1:]) for i in range(N)])
        if ref == 0 :
            Qout = Qin_c + np.array([np.sum(self.X[i:N]) for i in range(N)])
        else : 
            Qout = Qin_c + np.array([np.sum(self.X[:i+1]) for i in range(N)])

        uin = Qin/Ain
        ux = self.X/Ax
        uout = Qout/Aout
        Rein = fds.core.Reynolds(uin,Din,rho,mu=eta)
        Rex = fds.core.Reynolds(ux,Dx,rho,mu=eta)
        Lex = 0.05*np.sqrt(4*Ax/np.pi)*Rex
        Reout = fds.core.Reynolds(uout,Dout,rho,mu=eta)
        fin = [fds.friction.friction_factor(Re = Rein[i],eD = ep/Din) for i in range(N)]
        fx = [fds.friction.friction_factor(Re = Rex[i],eD=ep/Dx) for i in range(N)]
        fout = [fds.friction.friction_factor(Re = Reout[i],eD=ep/Dout) for i in range(N)]
        Kx_in = [Kxin(Din,Dx,theta,Qin[i],self.X[i],i,c_Kxin, method) for i in range(N)]
        Ky_in = [0]+[Kyin(Din,Dx,theta,Qin[i-1],self.X[i],i,c_Kyin, method=method) for i in range(1,N)]
        Kx_out = [Kxout(Dout,Dx,theta,Qout[i],self.X[i],i,c_Kxout, method=method) for i in range(N)]
        Ky_out = [Kyout(Dout,Dx,theta,Qout[i],self.X[i],i,c_Kyout, method=method) for i in range(N)]           

        field_df= pd.DataFrame({"Qin":Qin,"Qout":Qout,"uin":uin,"ux":ux,"uout":uout,"Rein":Rein,"Rex":Rex,"Reout":Reout,"fin":fin,"fx":fx,"fout":fout,"Kx_in":Kx_in,"Ky_in":Ky_in,"Kx_out":Kx_out,"Ky_out":Ky_out,"Lex":Lex})

        for column in field_df.columns:
            setattr(self, column, field_df[column])

class condition:
    def __init__(self, dic):
        """dic is a dictionary with the following keys:
        alpha, eta, rho, Qin_c, Qin_d
        """
        for key in dic.keys():
            setattr(self, key, dic[key])


def Kxin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"): 
    if method == 'Crane':
        return coeff*fds.fittings.K_branch_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return 1

def Kyin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"):
    if method == 'Crane':
        return coeff*fds.fittings.K_run_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return 1

def Kxout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"):
    if method == 'Crane':
        return coeff*fds.fittings.K_branch_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return 1

def Kyout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"):
    if method == 'Crane':
        return coeff*fds.fittings.K_run_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return 1


def inlet_pressure(field):
    N=field.par["N"]
    ref = field.par["ref"]
    Dx = field.par["D_riser"]
    Din = field.par["D_man"]
    Lx = field.par["L_riser"]
    Ly_list = field.par["Ly"]
    rho = field.cond.rho
    ux = field.ux
    uin = field.uin
    uout = field.uout
    Kxin = field.Kx_in
    Kxout = field.Kx_out
    Kyin = field.Ky_in
    fin = field.fin
    fx = field.fx
    Pin = np.zeros(N)
    Pin[ref] = (rho/2)*(fx[ref]*(Lx/Dx)*ux[ref]**2+Kxin[ref]*uin[ref]**2+Kxout[ref]*uout[ref]**2)
    if ref == 0:
        for i in range(1,N):
            Pin[i] = Pin[i-1]+(rho/2)*(uin[i-1]**2-uin[i]**2 + fin[i]*(Ly_list[i-1]/Din)*uin[i]**2 + Kyin[i]*uin[i]**2)
    else:
        for i in range(N-2,-1,-1):
            Pin[i] = Pin[i+1]+(rho/2)*(uin[i+1]**2-uin[i]**2 + fin[i]*(Ly_list[i+1]/Din)*uin[i]**2 + Kyin[i]*uin[i]**2)
    
    return Pin

def f(field):
    N = field.par["N"]
    Dx = field.par["D_riser"]
    Lx = field.par["L_riser"]
    rho = field.cond.rho
    ux = field.ux
    uin = field.uin
    uout = field.uout
    Kxin = field.Kx_in
    Kxout = field.Kx_out
    fx = field.fx
    Pout = np.zeros(N)
    Pin = inlet_pressure(field) 
    for i in range(N):
        Pout[i] = Pin[i] - (rho/2)*(fx[i]*(Lx/Dx)*ux[i]**2 + Kxin[i]*uin[i]**2 + Kxout[i]*uout[i]**2)

    return Pout

def g(field):
    N = field.par["N"]
    Dout = field.par["D_man"]
    rho = field.cond.rho
    uout = field.uout
    Ky_out = field.Ky_out
    Ly_list = field.par["Ly"]
    fout = field.fout
    Pout = np.zeros(N)
    if field.par["ref"] == 0:
        for i in range(1,N):
            Pout[i] = Pout[i-1] + (rho/2)*(uout[i-1]**2 - uout[i]**2 + fout[i]*(Ly_list[i-1]/Dout)*uout[i]**2 + Ky_out[i]*uout[i]**2)
    else:
        for i in range(N-2,-1,-1):
            Pout[i] = Pout[i+1] + (rho/2)*(uout[i+1]**2 - uout[i]**2 + fout[i]*(Ly_list[i+1]/Dout)*uout[i]**2 + Ky_out[i]*uout[i]**2)

    return Pout

def phi(field):
    X = field.X
    alpha = field.cond.alpha
    Qin_d = field.cond.Qin_d

    return np.sum(X)/alpha - Qin_d

def PL_fsolve(par, cond, q_init=[]):
    N = par["N"]
    if q_init == []:
        X0 = np.array([cond.Qin_d/N]*N)
    else:
        X0 = np.array(q_init)

    def fun(X):
        field = flow_field(X, par, cond)
        return abs(phi(field)) + np.linalg.norm(f(field) - g(field))

    Xsol = sc.minimize(fun,X0).x
    field = flow_field(Xsol, par, cond)
    tabl = pd.DataFrame({"Pin":inlet_pressure(field), "Pout":g(field), "qx":Xsol*3600000})
    tabl = tabl[::-1].reset_index(drop=True)

    return tabl, Xsol

