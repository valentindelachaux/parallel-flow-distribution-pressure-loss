import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import scipy.optimize as sc

# sch = exchanger or system

def Kxin(par,Qin_i,x,i,coeff): #Din, Dx, theta in par
    if par['method'] == 'Crane':
        return coeff*fds.fittings.K_branch_diverging_Crane(D_run=par['Din'], D_branch=par['Dx'], Q_run=Qin_i, Q_branch=x, angle=par['theta'])
    elif par['method'] == "input":
        return par['Kxin'][i]

def Kyin(par,Qout_im,x,i,coeff):
    if par['method'] == 'Crane':
        return coeff*fds.fittings.K_run_diverging_Crane(D_run=par['Din'], D_branch=par['Dx'], Q_run=Qout_im, Q_branch=x, angle=par['theta'])
    elif par['method'] == "input":
        return par['Kyin'][i]


def Kxout(par,Qout_i,x,i,coeff):
    if par['method'] == 'Crane':
        return coeff*fds.fittings.K_branch_converging_Crane(D_run=par['Dout'], D_branch=par['Dx'], Q_run=Qout_i, Q_branch=x, angle=par['theta'])
    elif par['method'] == "input":
        return par['Kxout'][i]

def Kyout(par,Qout_i,x,i,coeff):
    if par['method'] == 'Crane':
        return coeff*fds.fittings.K_run_converging_Crane(D_run=par['Dout'], D_branch=par['Dx'], Q_run=Qout_i, Q_branch=x, angle=par['theta'])
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

    c_Kxin = par["coeff_Kxin"]
    c_Kxout = par["coeff_Kxout"]
    c_Kyin = par["coeff_Kyin"]
    c_Kyout = par["coeff_Kyout"]

    Ly_list = par["Ly"] # N-1 values

    eta = par["eta"] # dynamic viscosity

    ep = par["rough"]

    rho = par["rho"]

    # Fonction = système de 3N équations

    def fun(x):

        leq = []
        for i in range(N):
            Qin_i = sum([x[j] for j in range(2*N,2*N+i+1)])
            Qout_i = sum([x[j] for j in range(2*N+i,3*N)])
            
            uin_i = Qin_i/Ain
            ux_i = x[2*N+i]/Ax
            uout_i = Qout_i/Aout

            Rein_i = fds.core.Reynolds(uin_i,Din,rho,mu=eta)
            Rex_i = fds.core.Reynolds(ux_i,Dx,rho,mu=eta)
            Reout_i = fds.core.Reynolds(uout_i,Dout,rho,mu=eta)
            fin_i = fds.friction.friction_factor(Re = Rein_i,eD = ep/Din)
            fx_i = fds.friction.friction_factor(Re = Rex_i,eD=ep/Dx)
            fout_i = fds.friction.friction_factor(Re = Reout_i,eD=ep/Dout)

            Kxin_i = Kxin(par,Qin_i,x[2*N+i],i,c_Kxin)
            if i >= 1:
                Qin_im = sum([x[j] for j in range(2*N,2*N+i)])
                Qout_im = sum([x[j] for j in range(2*N+i,3*N)])
                uin_im = Qin_im/Ain
                uout_im = Qout_im/Aout
                Kyin_i = Kyin(par,Qout_im,x[2*N+i],i,c_Kyin)
            Kxout_i = Kxout(par,Qout_i,x[2*N+i],i,c_Kxout)
            Kyout_i = Kyout(par,Qout_i,x[2*N+i],i,c_Kyout)

            if i>=1:
                leq.append(x[i] - x[i-1] - (rho/2)*(uin_im**2-uin_i**2 + (fin_i*Ly_list[i-1]*uin_i**2)/Din + Kyin_i*uin_i**2))
                leq.append(x[N+i] - x[N+i-1] - (rho/2)*(uout_im**2-uout_i**2 + (fout_i*Ly_list[i-1]*uout_i**2)/Dout + Kyout_i*uout_im**2))
            
            if sch == "exchanger":
                b_x = 0. # linear part
                a_x = fx_i*(Lx/Dx) # second order part
            elif sch == "system":
                a_x = par["a_x"]
                b_x = par["b_x"]
            leq.append(x[i] - x[N+i] - (rho/2)*(b_x*ux_i+a_x*ux_i**2+Kxin_i*uin_i**2+Kxout_i*uout_i**2))
        
        leq.append(sum([x[j] for j in range(2*N,3*N)]) - QF)
        leq.append(x[N+ref] - 0)

        return leq

    # Initialisation

    X0 = np.zeros(3*N)

    # Initialisation avec des débits uniformes
    for i in range(N):
        X0[2*N+i] = QF/N

    i = ref # ref = 0 or N-1
    Qin_ref = sum([X0[j] for j in range(2*N,2*N+i+1)])
    Qout_ref = sum([X0[j] for j in range(2*N+i,3*N)])

    uin_ref = Qin_ref/Ain
    ux_ref = X0[2*N+i]/Ax
    uout_ref = Qout_ref/Aout

    Rein_ref = fds.core.Reynolds(uin_ref,Din,rho,mu=eta)
    Rex_ref = fds.core.Reynolds(ux_ref,Dx,rho,mu=eta)
    Reout_ref = fds.core.Reynolds(uout_ref,Dout,rho,mu=eta)
    fin_ref = fds.friction.friction_factor(Re = Rein_ref)
    fx_ref = fds.friction.friction_factor(Re = Rex_ref)
    fout_ref = fds.friction.friction_factor(Re = Reout_ref)

    Kxin_ref = Kxin(par,Qin_ref,X0[2*N+i],ref,c_Kxin)
    Kxout_ref = Kxout(par,Qout_ref,X0[2*N+i],ref,c_Kxout)

    X0[N+ref] = 0
    DPx_ref = (rho/2)*(fx_ref*(Lx/Dx)*ux_ref**2+Kxin_ref*uin_ref**2+Kxout_ref*uout_ref**2)
    X0[ref] = DPx_ref+X0[N+ref]

    if ref == 0:
        ra = range(1,N)
    else:
        ra = range(0,N-1) 

    for i in ra:
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

        Kxin_i = Kxin(par,Qin_i,X0[2*N+i],i,c_Kxin)
        if i >= 1: # useless because the loop is for i = 1 to N-1 included
            Qin_im = sum([X0[j] for j in range(2*N,2*N+i+1)])
            Qout_im = sum([X0[j] for j in range(2*N+i,3*N)])
            uin_im = Qin_im/Ain
            uout_im = Qout_im/Aout
            Kyin_i = Kyin(par,Qout_im,X0[2*N+i],i,c_Kyin)
        Kxout_i = Kxout(par,Qout_i,X0[2*N+i],i,c_Kxout)
        Kyout_i = Kyout(par,Qout_i,X0[2*N+i],i,c_Kyout)

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

        X0[i] = X0[i-1] + (rho/2)*(uin_im**2-uin_i**2 + (fin_i*Ly_list[i-1]*uin_i**2)/Din + Kyin_i*uin_i**2)
        X0[N+i] = X0[N+i-1] + (rho/2)*(b_x*ux_i+a_x*ux_i**2+Kxin_i*uin_i**2+Kxout_i*uout_i**2)

    # Fin de l'initialisation

    Xsol = sc.fsolve(fun,X0)
    
    # Qin = []
    # Qout = []

    # for i in range(N):
    #     Qin.append(sum([Xsol[2*N+j]*3600000 for j in range(0,i)]))
    #     Qout.append(sum([Xsol[2*N+j]*3600000 for j in range(i,N)]))

    liste = [[Xsol[i],Xsol[N+i],Xsol[2*N+i]*3600000] for i in range(N)]

    df = pd.DataFrame(liste, columns = ['Pin','Pout','qx'])

    if print == True:
        display(HTML(df.to_html()))  

    return df,Xsol[N-1]