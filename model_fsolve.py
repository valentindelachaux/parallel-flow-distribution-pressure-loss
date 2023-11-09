import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import scipy.optimize as sc
import time as t


# sch = exchanger or system

def Kxin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,a=0,b=0,d=0,method="Crane"): #Din, Dx, theta in par
    if method == 'Crane':
        return (coeff+a*Q_run+b*Q_branch + d*Q_branch**2)*fds.fittings.K_branch_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kxin'][i]

def Kyin(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,a=0,b=0,d=0,method="Crane"):
    if method == 'Crane':
        return (coeff+a*Q_run+b*Q_branch+ d*Q_branch**2)*fds.fittings.K_run_diverging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kyin'][i]

def Kxout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,a=0,b=0,d=0,method="Crane"):
    if method == 'Crane':
        return (coeff+a*Q_run+b*Q_branch+ d*Q_branch**2)*fds.fittings.K_branch_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kxout'][i]

def Kyout(D_run,D_branch,theta,Q_run,Q_branch,i,coeff,a=0,b=0,d=0,method="Crane"):
    if method == 'Crane':
        return (coeff+a*Q_run+b*Q_branch+ d*Q_branch**2)*fds.fittings.K_run_converging_Crane(D_run=D_run, D_branch=D_branch, Q_run=Q_run, Q_branch=Q_branch, angle=theta)
    elif method == "input":
        return method['Kyout'][i]


def PL_fsolve_range(par,cond,list_Dv):
    list_PL = []
    list_tabl = []

    for Dv in list_Dv:
        cond["Dv"] = Dv
        tabl,res,PrL = PL_fsolve(par,cond)
        list_PL.append(res)
        list_tabl.append(tabl)

    return np.array(list_PL),list_tabl

def change_diameter(par, D, name='man'):
    former = par['D_'+ name]
    par['D_'+ name] = D
    par['A_' + name] *= (D/former)**2

def PL_fsolve_range_rd(par,cond,list_rd):
    list_PL = []
    list_tabl = []
    D0_riser = par['D_riser']
    D0_man = par['D_man']
    r0 = D0_riser/D0_man
    for rd in list_rd:
        change_diameter(par, D0_man*np.sqrt(r0/rd), name='man')
        change_diameter(par, D0_riser*np.sqrt(rd/r0), name='riser')
        tabl, res, PL, u, K = PL_fsolve(par,cond,show=False)
        list_PL.append(res)
        list_tabl.append(tabl)
    return np.array(list_PL),list_tabl


def calc(q_vect, par, cond, DR=1, series = False):
    """
    Args : 
        q_vect : list, débit volumique dans les canaux [m3/s]
        par
        cond

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
    # Parameters
    N=par["N"]
    eps = cond["eps"]
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
    Ly_list = par["Ly"]

    c_Kxin = par["coeff_Kxin"]
    c_Kxout = par["coeff_Kxout"]
    c_Kyin = par["coeff_Kyin"]
    c_Kyout = par["coeff_Kyout"]
    a = 0
    b = 0
    d = 0

    if series == False:
        QF_out = 0
        alpha = 1
    else :
        QF_out = cond["Dv_out"]
        alpha = cond["alpha"]

    eta = cond["eta"] # dynamic viscosity
    ep = par["roughness"]
    rho = cond["rho"]
    q_vect = np.array(q_vect)
    Qin = QF - np.array([np.sum(q_vect[i+1:]) for i in range(N)])
    
    if ref == 0 : # Z-type
        Qout = QF_out + np.array([np.sum(q_vect[i:N]) for i in range(N)])
    else : # U-type
        Qout = QF_out + np.array([np.sum(q_vect[:i+1]) for i in range(N)])

    uin = Qin/Ain
    ux = q_vect/Ax
    uout = Qout/Aout
    Rein = fds.core.Reynolds(uin,Din,rho,mu=eta)
    Rex = fds.core.Reynolds(ux,Dx,rho,mu=eta)
    Lex = 0.05*np.sqrt(4*Ax/np.pi)*Rex
    Reout = fds.core.Reynolds(uout,Dout,rho,mu=eta)
    fin = [fds.friction.friction_factor(Re = Rein[i],eD = ep/Din) for i in range(N)] # fonction non vectorisée
    fx = [fds.friction.friction_factor(Re = Rex[i],eD=ep/Dx) for i in range(N)]
    fout = [fds.friction.friction_factor(Re = Reout[i],eD=ep/Dout) for i in range(N)]
    Kx_in = [Kxin(Din,Dx,theta,Qin[i],q_vect[i],i,c_Kxin,a,b,d) for i in range(N)]
    if not(series):
        Ky_in = [0]+[Kyin(Din,Dx,theta,Qin[i-1],q_vect[i],i,c_Kyin,a,b,d) for i in range(1,N)]
    else :
        Ky_in = [Kyin(Din,Dx,theta,QF*(1-alpha),q_vect[0],0,c_Kyin,a,b,d)]+[Kyin(Din,Dx,theta,Qin[i-1],q_vect[i],i,c_Kyin,a,b,d) for i in range(1,N)]
    Kx_out = [Kxout(Dout,Dx,theta,Qout[i],q_vect[i],i,c_Kxout,a,b,d) for i in range(N)]
    Ky_out = [Kyout(Dout,Dx,theta,Qout[i],q_vect[i],i,c_Kyout,a,b,d) for i in range(N)]           
    Kse = fds.fittings.contraction_sharp(DR*Din, Din, fd=fin[N-1], Re=Rein[N-1], roughness=ep) 
    K_se = Kse*np.array([0.05, 0.1, 0.25, 0.6])
    K_se = np.concatenate((np.zeros(N-len(K_se)), K_se))

    return Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out, K_se, Lex

def initialize(q_init, par, cond, series = False):
    N = par["N"]
    Lx = par["L_riser"]
    Dx = par["D_riser"]
    Din = par["D_man"]
    ref = par["ref"] # 0 (en Z) ou N-1 (en U)
    QF = cond["Dv"]
    Ly_list = par["Ly"]

    if not(series):
        QF_out = 0
        alpha = 1
    else :
        QF_out = cond["Dv_out"]
        alpha = cond["alpha"]
    rho = cond["rho"]

    X0 = np.zeros(3*N)
    # Initialisation avec des débits uniformes
    if q_init == []:
        for i in range(N):
            X0[2*N+i] = QF*alpha/N
    else:
        for i in range(N):
            X0[2*N+i] = q_init[i]

    Qin_0, Qout_0, uin_0, ux_0, uout_0, Rein_0, Rex_0, Reout_0, fin_0, fx_0, fout_0, Kxin_0, Kyin_0, Kxout_0, Kyout_0, Kse_0, Lex_0 = calc(X0[2*N:], par, cond, series=series) 
    
    if par["sch"] == "exchanger":
            b_x = 0. # linear part
            a_x = fx_0[i]*(Lx/Dx) # second order part
    elif par["sch"] == "system":
            a_x = par["a_x"]
            b_x = par["b_x"]

    dPin_0 = [(rho/2)*(uin_0[i-1]**2-uin_0[i]**2 + (fin_0[i]*Ly_list[i-1]*uin_0[i]**2)/Din + Kyin_0[i]*uin_0[i]**2) for i in range(1,N-1)]
    DPx_ref = (rho/2)*(fx_0[ref]*(Lx/Dx)*ux_0[ref]**2+Kxin_0[ref]*uin_0[ref]**2+Kxout_0[ref]*uout_0[ref]**2)
    Pin_0 = [DPx_ref + sum(dPin_0[:i]) for i in range(N)]
    if ref == 0:
        dPout_0 = [(rho/2)*(b_x*ux_0[i]+fx_0[i]*(Lx/Dx)*ux_0[i]**2+Kxin_0[i]*uin_0[i]**2+Kxout_0[i]*uout_0[i]**2) for i in range(1,N)]
        Pout_0 = [sum(dPout_0[:i]) for i in range(N)]
    else :
        dPout_0 = [(rho/2)*(b_x*ux_0[i]+fx_0[i]*(Lx/Dx)*ux_0[i]**2+Kxin_0[i]*uin_0[i]**2+Kxout_0[i]*uout_0[i]**2) for i in range(N-1)]        
        Pout_0 = [- sum(dPout_0[i:]) for i in range(N)]
    X0[:N]=Pin_0
    X0[N:2*N]=Pout_0

    return X0

def compute_PL(q_sol, par, cond, series = False, fappx = 0.25):
    N = par["N"]
    Lx = par["L_riser"]
    Ly_list = par["Ly"]
    Dx = par["D_riser"]
    Din = par["D_man"]
    Dout = par["D_man"]
    ref = par["ref"] # 0 (en Z) ou N-1 (en U)
    rho = cond["rho"]

    if par["sch"] == "exchanger":
        b_x = 0. # linear part       
    elif par["sch"] == "system":
        b_x = par["b_x"]    

    Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out, K_se, Lex = calc(q_sol, par, cond, series=series)

    PRec = (rho/2)*(uin[N-1]**2-uout[ref]**2)*np.ones(N)
    PL_e = -(rho/2)*np.array([sum(K_se)*uin[N-1]**2 - sum([K_se[j]*uin[N-1]**2 for j in range(i+1,N)]) for i in range(N)])
    PL_riser = (rho/2)*np.array([b_x*ux[i]+((fx[i]*(Lx-Lex[i])+ 4*fappx*fx[i]*Lex[i])/Dx)*ux[i]**2 for i in range(N)])
    if ref==0:
        PL_t = (rho/2)*np.array([sum([Ky_in[j]*uin[j]**2 for j in range(i+1,N)]) + sum([Ky_out[j]*uout[j]**2 for j in range(0,i)]) + Kx_in[i]*uin[i]**2 + Kx_out[i]*uout[i]**2 for i in range(N)])
        PL_man = (rho/2)*np.array([sum([fin[j]*Ly_list[j-1]*uin[j]**2/Din for j in range(i+1,N)]) + sum([fout[j]*Ly_list[j-1]*uout[j]**2/Dout for j in range(1,i+1)]) for i in range(N)])

    else :
        PL_t = (rho/2)*np.array([sum([Ky_in[j]*uin[j]**2 for j in range(i+1,N)]) + sum([Ky_out[j]*uout[j]**2 for j in range(i+1,N)]) + Kx_in[i]*uin[i]**2 + Kx_out[i]*uout[i]**2 for i in range(N)])
        PL_man = (rho/2)*np.array([sum([fin[j]*Ly_list[j-1]*uin[j]**2/Din for j in range(i+1,N)]) + sum([fout[j]*Ly_list[j-1]*uout[j]**2/Dout for j in range(i+1,N)]) for i in range(N)])
    PL_tot = PL_e + PL_riser + PL_t + PL_man

    df_PL = pd.DataFrame((list(zip(PL_tot, PL_e, PL_man, PL_riser, PL_t, PRec))), columns = ["Total PL", "SPL entrance", "RPL manifold", "RPL riser", "SPL tee", "Pressure recovery"])
    df_PL = df_PL[::-1].reset_index(drop=True)

    return df_PL    

def PL_fsolve(par,cond, q_init=[],show=False, fappx = 0.25, DR = 1., series=False):

    # Parameters
    N = par["N"]
    QF = cond["Dv"]
    if not(series):
        QF_out = 0
        alpha = 1
    else :
        QF_out = cond["Dv_out"]
        alpha =cond["alpha"]
    ref = par["ref"] # 0 (en Z) ou N-1 (en U)

    Dx = par["D_riser"]
    Din = par["D_man"]
    Dout = par["D_man"]
    Lx = par["L_riser"]

    Ly_list = par["Ly"] # N-1 values
    rho = cond["rho"]

    # Fonction = système de 3N équations    

    def fun(x):
        """
        Args : 
            x : list, system state in the format [Pin_0, ..., Pin_N-1, Pout_0, ..., Pout_N-1, q_0, ..., q_N-1]
            
        Returns : 
            leq : list, system of equations to which x is subjected
        """ 
        leq = []
        Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out, K_se, Lex = calc(x[2*N:3*N], par, cond, DR, series)

        
        for i in range(N):
            if i>=1:
                leq.append(x[i] - x[i-1] - (rho/2)*(uin[i-1]**2-uin[i]**2 + (fin[i]*Ly_list[i-1]*uin[i]**2)/Din + Ky_in[i]*uin[i-1]**2 + K_se[i] * uin[N-1]**2))
                if ref == 0 :
                    leq.append(x[N+i] - x[N+i-1] - (rho/2)*(uout[i-1]**2-uout[i]**2 + (fout[i]*Ly_list[i-1]*uout[i]**2)/Dout + Ky_out[i]*uout[i-1]**2))
                else :
                    leq.append(x[N+i-1] - x[N+i] - (rho/2)*(uout[i]**2-uout[i-1]**2 + (fout[i]*Ly_list[i-1]*uout[i-1]**2)/Dout + Ky_out[i]*uout[i]**2)) 
                                
            if par["sch"] == "exchanger":
                b_x = 0. # linear part
                a_x = (fx[i]*(Lx-Lex[i])+ 4*fappx*fx[i]*Lex[i])/Dx  # second order part
                        
            elif par["sch"] == "system":
                a_x = par["a_x"]
                b_x = par["b_x"]
            leq.append(x[i] - x[N+i] - (rho/2)*(uout[i]**2-uin[i]**2 + b_x*ux[i]+a_x*ux[i]**2+Kx_in[i]*uin[i]**2+Kx_out[i]*uout[i]**2))

        leq.append(sum([x[j] for j in range(2*N,3*N)]) - QF*alpha)
        leq.append(x[N+ref] - 0)
        return(leq)

    # Initialisation

    X0 = initialize(q_init, par, cond, series)

    t1 = t.time()
    Xsol = sc.fsolve(fun,X0)
    t2 = t.time()
    print(t2-t1)

    liste = [[Xsol[i],Xsol[N+i],Xsol[2*N+i]*3600000] for i in range(N)]
    df = pd.DataFrame(liste, columns = ['Pin','Pout','qx'])
    df = df[::-1].reset_index(drop=True)

    df_PL = compute_PL(Xsol[2*N:3*N], par, cond, series, fappx)

    # df_u = pd.DataFrame((list(zip(uin,ux,uout))), columns=['uin', 'ux', 'uout'])
    # df_u = df_u[::-1].reset_index(drop=True)
    # df_K = pd.DataFrame((list(zip(Kx_in, Ky_in, Kx_out, Ky_out))), columns=['Kx_in', 'Ky_in', 'Kx_out', 'Ky_out'])
    # df_K = df_K[::-1].reset_index(drop=True)

    if print == True:
        display(HTML(df.to_html()))  

    return df,Xsol[N-1], df_PL



