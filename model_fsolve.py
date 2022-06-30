import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import scipy.optimize as sc

# sch = exchanger or system

def Kxin(par,Qin_i,x,i): #Din, Dx, theta in par
    if par['method'] == 'Crane':
        return fds.fittings.K_branch_diverging_Crane(D_run=par['Din'], D_branch=par['Dx'], Q_run=Qin_i, Q_branch=x, angle=par['theta'])
    elif par['method'] == "input":
        return par['Kxin'][i]

def Kyin(par,Qout_im,x,i):
    if par['method'] == 'Crane':
        return fds.fittings.K_run_diverging_Crane(D_run=par['Din'], D_branch=par['Dx'], Q_run=Qout_im, Q_branch=x, angle=par['theta'])
    elif par['method'] == "input":
        return par['Kyin'][i]


def Kxout(par,Qout_i,x,i):
    if par['method'] == 'Crane':
        return fds.fittings.K_branch_converging_Crane(D_run=par['Dout'], D_branch=par['Dx'], Q_run=Qout_i, Q_branch=x, angle=par['theta'])
    elif par['method'] == "input":
        return par['Kxout'][i]

def Kyout(par,Qout_i,x,i):
    if par['method'] == 'Crane':
        return fds.fittings.K_run_converging_Crane(D_run=par['Dout'], D_branch=par['Dx'], Q_run=Qout_i, Q_branch=x, angle=par['theta'])
    elif par['method'] == "input":
        return par['Kyout'][i]

def PL_fsolve(par,sch,print):

    # Parameters
    eps = par["eps"]

    N = par["N"]
    QF = par["QF"]
    U = par["U"]

    ref = par["ref"]

    Ax = par["Ax"]
    Ain = par["Ain"]
    Aout = par["Aout"]
    Dx = par["Dx"]
    Din = par["Din"]
    Dout = par["Dout"]
    Lx = par["Lx"]
    Ly = par["Ly"]
    eta = par["eta"] # dynamic viscosity

    ep = par["rough"]

    rho = par["rho"]

    # Fonction

    def fun(x):

        leq = []
        for i in range(N):
            Qin_i = sum([x[j] for j in range(2*N,2*N+i+1)])
            Qout_i = sum([x[j] for j in range(2*N+i,3*N)])
            
            uin_i = Qin_i/Ain
            ux_i = x[2*N+i]/Ax
            uout_i = Qout_i/Aout

            Rein_i = fds.core.Reynolds(uin_i,Din,rho,mu=eta)
            Rex_i = fds.core.Reynolds(ux_i,Din,rho,mu=eta)
            Reout_i = fds.core.Reynolds(uout_i,Din,rho,mu=eta)
            fin_i = fds.friction.friction_factor(Re = Rein_i,eD = ep/Din)
            fx_i = fds.friction.friction_factor(Re = Rex_i,eD=ep/Dx)
            fout_i = fds.friction.friction_factor(Re = Reout_i,eD=ep/Dout)

            Kxin_i = Kxin(par,Qin_i,x[2*N+i],i)
            if i >= 1:
                Qin_im = sum([x[j] for j in range(2*N,2*N+i)])
                Qout_im = sum([x[j] for j in range(2*N+i,3*N)])
                uin_im = Qin_im/Ain
                uout_im = Qout_im/Aout
                Kyin_i = Kyin(par,Qout_im,x[2*N+i],i)
            Kxout_i = Kxout(par,Qout_i,x[2*N+i],i)
            Kyout_i = Kyout(par,Qout_i,x[2*N+i],i)

            if i>=1:
                leq.append(x[i] - x[i-1] - (rho/2)*(uin_im**2-uin_i**2 + (fin_i*Ly*uin_i**2)/Din + Kyin_i*uin_i**2))
                leq.append(x[N+i] - x[N+i-1] - (rho/2)*(uout_im**2-uout_i**2 + (fout_i*Ly*uout_i**2)/Dout + Kyout_i*uout_im**2))
            
            if sch == "exchanger":
                b_x = 0. # linear part
                a_x = fx_i*(Lx/Dx) # second order part
            elif sch == "system":
                a_x = par["a_x"]
                b_x = par["b_x"]
            leq.append(x[i] - x[N+i] - (rho/2)*(b_x*ux_i+a_x*ux_i**2+Kxin_i*uin_i**2+Kxout_i*uout_i**2))
        
        leq.append(sum([x[j] for j in range(2*N,3*N)]) - QF)
        leq.append(x[N] - 0)

        return leq

    # Initialisation

    X0 = np.zeros(3*N)

    for i in range(N):
        X0[2*N+i] = QF/N

    i = 0
    Qin_0 = sum([X0[j] for j in range(2*N,2*N+i+1)])
    Qout_0 = sum([X0[j] for j in range(2*N+i,3*N)])

    uin_0 = Qin_0/Ain
    ux_0 = X0[2*N+i]/Ax
    uout_0 = Qout_0/Aout

    Rein_0 = fds.core.Reynolds(uin_0,Din,rho,mu=eta)
    Rex_0 = fds.core.Reynolds(ux_0,Dx,rho,mu=eta)
    Reout_0 = fds.core.Reynolds(uout_0,Dout,rho,mu=eta)
    fin_0 = fds.friction.friction_factor(Re = Rein_0)
    fx_0 = fds.friction.friction_factor(Re = Rex_0)
    fout_0 = fds.friction.friction_factor(Re = Reout_0)

    Kxin_0 = Kxin(par,Qin_0,X0[2*N+i],0)
    Kxout_0 = Kxout(par,Qout_0,X0[2*N+i],0)

    X0[N+0] = 0
    DPx_0 = (rho/2)*(fx_0*(Lx/Dx)*ux_0**2+Kxin_0*uin_0**2+Kxout_0*uout_0**2)
    X0[0] = DPx_0+X0[N+0]

    for i in range(1,N):
        Qin_i = sum([X0[j] for j in range(2*N,2*N+i)])
        Qout_i = sum([X0[j] for j in range(2*N+i,3*N)])
        
        uin_i = Qin_i/Ain
        ux_i = X0[2*N+i]/Ax
        uout_i = Qout_i/Aout

        Rein_i = fds.core.Reynolds(uin_i,Din,rho,mu=eta)
        Rex_i = fds.core.Reynolds(ux_i,Dx,rho,mu=eta)
        Reout_i = fds.core.Reynolds(uout_i,Dout,rho,mu=eta)
        fin_i = fds.friction.friction_factor(Re = Rein_i)
        fx_i = fds.friction.friction_factor(Re = Rex_i)
        fout_i = fds.friction.friction_factor(Re = Reout_i)

        Kxin_i = Kxin(par,Qin_i,X0[2*N+i],i)
        if i >= 1: # useless because the loop is for i = 1 to N-1 included
            Qin_im = sum([X0[j] for j in range(2*N,2*N+i+1)])
            Qout_im = sum([X0[j] for j in range(2*N+i,3*N)])
            uin_im = Qin_im/Ain
            uout_im = Qout_im/Aout
            Kyin_i = Kyin(par,Qout_im,X0[2*N+i],i)
        Kxout_i = Kxout(par,Qout_i,X0[2*N+i],i)
        Kyout_i = Kyout(par,Qout_i,X0[2*N+i],i)

        if sch == "exchanger":
            pass
        elif sch == "system":
            b_x = 0. # linear part
            a_x = fx_i*(Lx/Dx) # second order part

        if sch == "exchanger":
            b_x = 0. # linear part
            a_x = fx_i*(Lx/Dx) # second order part
        elif sch == "system":
            a_x = par["a_x"]
            b_x = par["b_x"]

        X0[i] = X0[i-1] + (rho/2)*(uin_im**2-uin_i**2 + (fin_i*Ly*uin_i**2)/Din + Kyin_i*uin_i**2)
        X0[N+i] = X0[N+i-1] + (rho/2)*(b_x*ux_i+a_x*ux_i**2+Kxin_i*uin_i**2+Kxout_i*uout_i**2)

    # Fin de l'initialisation

    Xsol = sc.fsolve(fun,X0)
    
    # Qin = []
    # Qout = []

    # for i in range(N):
    #     Qin.append(sum([Xsol[2*N+j]*3600000 for j in range(0,i)]))
    #     Qout.append(sum([Xsol[2*N+j]*3600000 for j in range(i,N)]))

    liste = [[Xsol[i],Xsol[N+i],Xsol[2*N+i]*3600000] for i in range(N)]

    df = pd.DataFrame(liste, columns = ['Pin','Pout','qx','Qin','Qout'])

    if print == True:
        display(HTML(df.to_html()))  

    return df,Xsol[N-1]