import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import scipy.optimize as sc

# sch = exchanger or system

def Kxin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"): #Din, Dx, theta in par
    if method == 'Crane':
        return coeff*fds.fittings.K_branch_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kxin'][i]

def Kyin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"):
    if method == 'Crane':
        return coeff*fds.fittings.K_run_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kyin'][i]

def Kxout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"):
    if method == 'Crane':
        return coeff*fds.fittings.K_branch_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kxout'][i]

def Kyout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,method="Crane"):
    if method == 'Crane':
        return coeff*fds.fittings.K_run_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kyout'][i]

def PL_fsolve_range(par,cond,list_Dv):
    list_PL = []
    list_tabl = []
    list_mn = []
    list_std = []

    for Dv in list_Dv:
        print(Dv)
        cond["Dv"] = Dv

        tabl,res = PL_fsolve(par,cond)
        list_PL.append(res)
        list_tabl.append(tabl)

        list_mn.append(tabl['qx'].mean()) # fow rate qx is in L/h
        list_std.append(tabl['qx'].std())
    
    return list_PL,list_tabl,list_mn,list_std


def PL_fsolve(par,cond,q_init=[],print=False):

    # Parameters
    eps = cond["eps"]

    N = par["N"]
    QF = cond["Dv"]

    ref = par["ref"] # 0 (en Z) ou N-1 (en U)
    theta = par["theta"]

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

    # Fonction = système de 3N équations
    def calc(q_vect):
        """
        Args : 
            q_vect : list, débits dans les canaux

        Returns :
            Qin : list, débits manifold inlet
            Qout : list, débits manifold outlet
            uin : list, vitesses manifold intlet
            ux : list, vitesses canaux
            uout : list, vitesses manifold outlet
            Rein : list, nombre de Reynolds manifold inlet
            Rex : list, nombre de Reynolds canaux
            Reout : list, nombre de Reynolds manifold outlet
            fin : list, coefficient pertes de charges régulières manifold inlet
            fx : list, coefficient pertes de charges régulières canaux
            fout : list, coefficient pertes de charges régulières manifold outlet
            Kxin : list, coefficient pertes de charges singulières t manifold inlet selon axe x
            Kyin : list, coefficient pertes de charges singulières t manifold inlet selon axe y
            Kxout : list, coefficient pertes de charges singulières t manifold outlet selon axe x
            Kyout : list, coefficient pertes de charges singulières t manifold outlet selon axe y
        """
    
        q_vect = np.array(q_vect)
        Qin = np.array([np.sum(q_vect[:i+1]) for i in range(N)])
        
        if ref == 0 : # Z-type
            Qout = np.array([np.sum(q_vect[i:N]) for i in range(N)])
        else : # U-type
            Qout = Qin.copy()

        uin = Qin/Ain
        ux = q_vect/Ax
        uout = Qout/Aout
        Rein = fds.core.Reynolds(uin,Din,rho,mu=eta)
        Rex = fds.core.Reynolds(ux,Dx,rho,mu=eta)
        Reout = fds.core.Reynolds(uout,Dout,rho,mu=eta)
        fin = [fds.friction.friction_factor(Re = Rein[i],eD = ep/Din) for i in range(N)] # fonction non vectorisée
        fx = [fds.friction.friction_factor(Re = Rex[i],eD=ep/Dx) for i in range(N)]
        fout = [fds.friction.friction_factor(Re = Reout[i],eD=ep/Dout) for i in range(N)]
        Kx_in = [Kxin(Din,Dx,theta,Qin[i],q_vect[i],i,c_Kxin) for i in range(N)]
        Ky_in = [0]+[Kyin(Din,Dx,theta,Qin[i-1],q_vect[i],i,c_Kyin) for i in range(1,N)]
        Kx_out = [Kxout(Dout,Dx,theta,Qout[i],q_vect[i],i,c_Kxout) for i in range(N)]
        Ky_out = [Kyout(Dout,Dx,theta,Qout[i],q_vect[i],i,c_Kyout) for i in range(N)]           

        return Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out
    

    def fun(x):
        """
        Args : 
            x : list, system state in the format [Pin_0, ..., Pin_N-1, Pout_0, ..., Pout_N-1, q_0, ..., q_N-1]
            
        Returns : 
            leq : list, system of equations to which x is subjected
        """ 
        leq = []
        Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out = calc(x[2*N:3*N])

        for i in range(N):
            if i>=1:
                    
                leq.append(x[i] - x[i-1] - (rho/2)*(uin[i-1]**2-uin[i]**2 + (fin[i]*Ly_list[i-1]*uin[i]**2)/Din + Ky_in[i]*uin[i]**2))

                if ref == 0 :
                    leq.append(x[N+i] - x[N+i-1] - (rho/2)*(uout[i-1]**2-uout[i]**2 + (fout[i]*Ly_list[i-1]*uout[i]**2)/Dout + Ky_out[i]*uout[i-1]**2))
                else :
                    leq.append(x[N+i-1] - x[N+i] - (rho/2)*(uout[i]**2-uout[i-1]**2 + (fout[i]*Ly_list[i-1]*uout[i-1]**2)/Dout + Ky_out[i]*uout[i]**2)) 
                                
            if par["sch"] == "exchanger":
                b_x = 0. # linear part
                a_x = fx[i]*(Lx/Dx) # second order part
            elif par["sch"] == "system":
                a_x = par["a_x"]
                b_x = par["b_x"]
            leq.append(x[i] - x[N+i] - (rho/2)*(b_x*ux[i]+a_x*ux[i]**2+Kx_in[i]*uin[i]**2+Kx_out[i]*uout[i]**2))

        leq.append(sum([x[j] for j in range(2*N,3*N)]) - QF)
        leq.append(x[N+ref] - 0)
        return(leq)

    # Initialisation

    X0 = np.zeros(3*N)

    # Initialisation avec des débits uniformes
    if q_init == []:
        for i in range(N):
            X0[2*N+i] = QF/N
    else:
        for i in range(N):
            X0[2*N+i] = q_init[i]

    Qin_0, Qout_0, uin_0, ux_0, uout_0, Rein_0, Rex_0, Reout_0, fin_0, fx_0, fout_0, Kxin_0, Kyin_0, Kxout_0, Kyout_0 = calc(X0[2*N:]) 

    if par["sch"] == "exchanger":
            b_x = 0. # linear part
            a_x = fx_0[i]*(Lx/Dx) # second order part
    elif par["sch"] == "system":
            a_x = par["a_x"]
            b_x = par["b_x"]


    dPin_0 = [(rho/2)*(uin_0[i-1]**2-uin_0[i]**2 + (fin_0[i]*Ly_list[i-1]*uin_0[i]**2)/Din + Kyin_0[i]*uin_0[i]**2) for i in range(1,N)]
    DPx_ref = (rho/2)*(fx_0[ref]*(Lx/Dx)*ux_0[ref]**2+Kxin_0[ref]*uin_0[ref]**2+Kxout_0[ref]*uout_0[ref]**2)
    Pin_0 = [DPx_ref + sum(dPin_0[:i]) for i in range(N)]

    if ref == 0:
        dPout_0 = [(rho/2)*(b_x*ux_0[i]+a_x*ux_0[i]**2+Kxin_0[i]*uin_0[i]**2+Kxout_0[i]*uout_0[i]**2) for i in range(1,N)]
        Pout_0 = [sum(dPout_0[:i]) for i in range(N)]

    else :
        dPout_0 = [(rho/2)*(b_x*ux_0[i]+a_x*ux_0[i]**2+Kxin_0[i]*uin_0[i]**2+Kxout_0[i]*uout_0[i]**2) for i in range(N-1)]        
        Pout_0 = [- sum(dPout_0[i:]) for i in range(N)]

    X0[:N]=Pin_0
    X0[N:2*N]=Pout_0

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