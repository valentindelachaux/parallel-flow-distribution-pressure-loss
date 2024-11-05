import pandas as pd
import numpy as np
import fluids as fds
import scipy.optimize as sc

### Tés ###
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

### Résolution pour un échangeur et un set de conditions ###
def calc(q_vect, par, cond, series = False):
    """
    Calcule les paramètres hydrauliques du système (débits, vitesses, pertes de charges, Reynolds, ...) à partir des débits dans les canaux

    Args : 
        q_vect : list, débit volumique dans les canaux [m3/s]
        par
        cond
        series (False ou [QF, QFout, alpha]) : permet de créer les abaques PL = f(QF, QFout, alpha)

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
        Lex : list, longueur de développement de l'écoulement dans les canaux
    """
    # Parameters
    N = par["N"]

    ref = par["ref"] # 0 (en Z) ou N-1 (en U)
    theta = par["theta"]

    Ax = par["A_riser"]
    Ain = par["A_man"]
    Aout = par["A_man"]
    Dx = par["D_riser"]
    D_in = par["D_in"]
    D_out = par["D_out"]

    c_Kxin = par["coeff_Kxin"]
    c_Kxout = par["coeff_Kxout"]
    c_Kyin = par["coeff_Kyin"]
    c_Kyout = par["coeff_Kyout"]

    method = par["method"]

    if not(series):
        QF = cond["Dv"]
        QF_out = 0
        alpha = 1
    else :
        QF = series[0]
        QF_out = series[1]
        alpha = series[2]

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
    Rein = fds.core.Reynolds(uin,D_in,rho,mu=eta)
    Rex = fds.core.Reynolds(ux,Dx,rho,mu=eta)
    Lex = 0.05*np.sqrt(4*Ax/np.pi)*Rex
    Reout = fds.core.Reynolds(uout,D_out,rho,mu=eta)
    fin = [fds.friction.friction_factor(Re = Rein[i],eD = ep/D_in) for i in range(N)]

    if par['specific_inter_panel'] == 1:
        for i in range(N):
            if (i >= 1) & (i % (par['N_riser_per_panel']) == 0):
                fin[i] = par['inter_panel_coeff'] * D_in / (par['Ly'][i-1])

    fx = [fds.friction.friction_factor(Re = Rex[i],eD=ep/Dx) for i in range(N)]
    fout = [fds.friction.friction_factor(Re = Reout[i],eD=ep/D_out) for i in range(N)]
    Kx_in = [Kxin(D_in,Dx,theta,Qin[i],q_vect[i],i,c_Kxin, method) for i in range(N)]
    if not(series):
        Ky_in = [0]+[Kyin(D_in,Dx,theta,Qin[i-1],q_vect[i],i,c_Kyin, method=method) for i in range(1,N)]
    else :
        Ky_in = [Kyin(D_in,Dx,theta,QF*(1-alpha),q_vect[0],0,c_Kyin, method=method)]+[Kyin(D_in,Dx,theta,Qin[i-1],q_vect[i],i,c_Kyin, method=method) for i in range(1,N)]
    Kx_out = [Kxout(D_out,Dx,theta,Qout[i],q_vect[i],i,c_Kxout, method=method) for i in range(N)]
    Ky_out = [Kyout(D_out,Dx,theta,Qout[i],q_vect[i],i,c_Kyout, method=method) for i in range(N)]           
    
    return Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out, Lex

def initialize(q_init, par, cond, series = False):
    """
    Etat initial (pressions et débits dans l'échangeur) avant optimisation

    Args :
        q_init : list, débits initiaux dans les canaux [m3/s] ou liste vide pour un initialisation homogène
        par : dictionnaire des paramètres de l'échangeur
        cond : dictionnaire des conditions de l'étude (paramètres du fluide)
        series (False ou [QF, QFout, alpha]) : permet d'indiquer si l'échangeur dont on calcule les pertes de charges est en série avec d'autres ou non
    
    Returns :
        X0 : list, état initial du système (pressions et débits dans l'échangeur)"""
    
    N = par["N"]
    Lx = par["L_riser"]
    Dx = par["D_riser"]
    D_in = par["D_in"]
    ref = par["ref"] # 0 (en Z) ou N-1 (en U)
    Ly_list = par["Ly"]

    if not(series):
        QF = cond["Dv"]
        alpha = 1
    else :
        QF = series[0]
        alpha = series[2]
    rho = cond["rho"]

    X0 = np.zeros(3*N)
    # Initialisation avec des débits uniformes
    if q_init == []:
        for i in range(N):
            X0[2*N+i] = QF*alpha/N
    else:
        for i in range(N):
            X0[2*N+i] = q_init[i]

    Qin_0, Qout_0, uin_0, ux_0, uout_0, Rein_0, Rex_0, Reout_0, fin_0, fx_0, fout_0, Kxin_0, Kyin_0, Kxout_0, Kyout_0, Lex_0 = calc(X0[2*N:], par, cond, series=series) 
    
    if par["regular_PL_calculation_method"] == "friction_factor":
            b_x = 0. # linear part
            a_x = fx_0[i]*(Lx/Dx) # second order part
    elif par["regular_PL_calculation_method"] == "custom":
            a_x = par["a_x"]
            b_x = par["b_x"]

    dPin_0 = [(rho/2)*(uin_0[i-1]**2-uin_0[i]**2 + (fin_0[i]*Ly_list[i-1]*uin_0[i]**2)/D_in + Kyin_0[i]*uin_0[i]**2) for i in range(1,N-1)]
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

    return X0

def compute_PL(q_sol, par, cond, series = False, fappx = 0.25):
    """
    Calcule les pertes de charges selon chacune des causes pour chacun des trajets empruntés par le fluide dans l'échangeur

    Args :
        q_sol : list, débits dans les canaux [m3/s]
        par : dictionnaire des paramètres de l'échangeur
        cond : dictionnaire des conditions de l'étude (paramètres du fluide)
        series (False ou [QF, QFout, alpha]) : permet d'indiquer si l'échangeur dont on calcule les pertes de charges est en série avec d'autres échangeurs ou seul
        fappx : float, 1/4 du facteur multiplicatif des pertes de charges sur la longueur des développement de l'écoulement dans les canaux

    Returns :
        df_PL : dataframe, pertes de charges selon chacune des causes
    """
    
    N = par["N"]
    Lx = par["L_riser"]
    Ly_list = par["Ly"]
    Dx = par["D_riser"]
    D_in = par["D_in"]
    D_out = par["D_out"]
    ref = par["ref"]
    rho = cond["rho"]

    Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out, Lex = calc(q_sol, par, cond, series=series)

    if par["regular_PL_calculation_method"] == "friction_factor":
        b_x = np.zeros(N) # linear part
        a_x = np.array([(fx[i]*(Lx-Lex[i])+ 4*fappx*fx[i]*Lex[i])/Dx for i in range(N)])  # second order part
                
    elif par["regular_PL_calculation_method"] == "custom":
        a_x = np.array([par["a_x"] for i in range(N)])
        b_x = np.array([par["b_x"] for i in range(N)])

    elif par["regular_PL_calculation_method"] == "Triple":
        a_x = Triple_ax(x[2*N+i])
        b_x = 0.

    PRec = (rho/2)*(uin[N-1]**2-uout[ref]**2)*np.ones(N)
    PL_riser = (rho/2)*np.array([ b_x[i]*ux[i] + a_x[i]*ux[i]**2 for i in range(N)])

    if ref==0:
        PL_t = (rho/2)*np.array([sum([Ky_in[j]*uin[j]**2 for j in range(i+1,N)]) + sum([Ky_out[j]*uout[j]**2 for j in range(0,i)]) + Kx_in[i]*uin[i]**2 + Kx_out[i]*uout[i]**2 for i in range(N)])
        PL_man = (rho/2)*np.array([sum([fin[j]*Ly_list[j-1]*uin[j]**2/D_in for j in range(i+1,N)]) + sum([fout[j]*Ly_list[j-1]*uout[j]**2/D_out for j in range(1,i+1)]) for i in range(N)])

    else :
        PL_t = (rho/2)*np.array([sum([Ky_in[j]*uin[j]**2 for j in range(i+1,N)]) + sum([Ky_out[j]*uout[j]**2 for j in range(i+1,N)]) + Kx_in[i]*uin[i]**2 + Kx_out[i]*uout[i]**2 for i in range(N)])
        PL_man = (rho/2)*np.array([sum([fin[j]*Ly_list[j-1]*uin[j]**2/D_in for j in range(i+1,N)]) + sum([fout[j]*Ly_list[j-1]*uout[j]**2/D_out for j in range(i+1,N)]) for i in range(N)])
    
    PL_tot = PL_riser + PL_t + PL_man
    df_PL = pd.DataFrame((list(zip(PL_tot,  PL_man, PL_riser, PL_t, PRec))), columns = ["Total PL", "RPL manifold", "RPL riser", "SPL tee", "Pressure recovery"])
    df_PL = df_PL[::-1].reset_index(drop=True)

    return df_PL    

def PL_fsolve(par,cond, q_init=[], fappx = 0.25, series=False, maxfev=0):
    """
    Résout le système d'équations pour trouver le champ des pressions et débits dans l'échangeur
    
    Args :
        par : dict, dictionnaire des paramètres de l'échangeur
        cond : dict, dictionnaire des conditions de l'étude (paramètres du fluide)
        q_init : list, débits initiaux dans les canaux [m3/s] ou liste vide pour un initialisation homogène
        fappx : float, 1/4 du facteur multiplicatif des pertes de charges sur la longueur des développement de l'écoulement dans les canaux
        series (False ou [QF, QFout, alpha]) : permet d'indiquer si l'échangeur dont on calcule les pertes de charges est en série avec d'autres ou non

    Returns :
        df : dataframe, état du système (pressions et débits dans l'échangeur)
        PL : float, valeur des pertes de charges totales entrée/sortie de l'échangeur
        df_PL : dataframe, pertes de charges selon chacune des causes possibles et selon le trajet emprunté par le fluide dans l'échangeur
        residuals : list, résidus de l'optimisation
    """
    # Parameters
    N = par["N"]

    if not(series):
        QF = cond["Dv"]
        alpha = 1
    else :
        QF = series[0]
        alpha = series[2]
    ref = par["ref"] # 0 (en Z) ou N-1 (en U)

    Dx = par["D_riser"]
    D_in = par["D_in"]
    D_out = par["D_out"]
    Lx = par["L_riser"]

    Ly_list = par["Ly"] # N-1 values
    rho = cond["rho"]

    # Fonction = système de 3N équations    

    def fun(x):

        leq = []

        Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out, Lex = calc(x[2*N:3*N], par, cond, series)

        for i in range(N):
            if i>=1:
                leq.append(x[i] - x[i-1] - (rho/2)*(uin[i-1]**2-uin[i]**2 + (fin[i]*Ly_list[i-1]*uin[i]**2)/D_in + Ky_in[i]*uin[i-1]**2))
                if ref == 0 :
                    leq.append(x[N+i] - x[N+i-1] - (rho/2)*(uout[i-1]**2-uout[i]**2 + (fout[i]*Ly_list[i-1]*uout[i]**2)/D_out + Ky_out[i-1]*uout[i-1]**2))
                else :
                    leq.append(x[N+i-1] - x[N+i] - (rho/2)*(uout[i]**2-uout[i-1]**2 + (fout[i]*Ly_list[i-1]*uout[i-1]**2)/D_out + Ky_out[i]*uout[i]**2)) 
                                
            if par["regular_PL_calculation_method"] == "friction_factor":
                b_x = 0. # linear part
                a_x = (fx[i]*(Lx-Lex[i])+ 4*fappx*fx[i]*Lex[i])/Dx  # second order part
                        
            elif par["regular_PL_calculation_method"] == "custom":
                a_x = par["a_x"]
                b_x = par["b_x"]

            elif par["regular_PL_calculation_method"] == "Triple":
                a_x = Triple_ax(x[2*N+i])
                b_x = 0.

            leq.append(x[i] - x[N+i] - (rho/2)*(uout[i]**2-uin[i]**2 + b_x*ux[i] + a_x*ux[i]**2 + Kx_in[i]*uin[i]**2 + Kx_out[i]*uout[i]**2))
        
        leq.append(sum([x[j] for j in range(2*N,3*N)]) - QF*alpha)
        leq.append(x[N+ref] - 0)
        return(leq)

    # Initialisation

    X0 = initialize(q_init, par, cond, series)

    Xsol = sc.fsolve(fun, X0, maxfev=maxfev)

    Qin, Qout, uin, ux, uout, Rein, Rex, Reout, fin, fx, fout, Kx_in, Ky_in, Kx_out, Ky_out, Lex = calc(Xsol[2*N:3*N], par, cond, series)

    liste = [[Xsol[i],Xsol[N+i],Xsol[2*N+i]*3600000] for i in range(N)]
    df = pd.DataFrame(liste, columns = ['Pin','Pout','qx'])
    df = df[::-1].reset_index(drop=True)

    df_PL = compute_PL(Xsol[2*N:3*N], par, cond, series, fappx)

    df_u = pd.DataFrame((list(zip(uin,ux,uout))), columns=['uin', 'ux', 'uout'])
    df_u = df_u[::-1].reset_index(drop=True)

    df_K = pd.DataFrame((list(zip(Kx_in, Ky_in, Kx_out, Ky_out))), columns=['Kx_in', 'Ky_in', 'Kx_out', 'Ky_out'])
    df_K = df_K[::-1].reset_index(drop=True)

    df_Re = pd.DataFrame((list(zip(Rein, Rex, Reout))), columns=['Rein', 'Rex', 'Reout'])
    df_Re = df_Re[::-1].reset_index(drop=True)

    df_f = pd.DataFrame((list(zip(fin, fx, fout))), columns=['fin', 'fx', 'fout'])

    df_PL = pd.concat([df_PL, df_u, df_Re, df_K, df_f], axis=1)

    return df, Xsol[N-1- ref], df_PL, fun(Xsol)

### Résolutions pour des range de conditions ###
def PL_fsolve_range(par,cond,list_Dv, fappx=0.25):
    """
    Calcule les pertes de charges totales entrée/sortie de l'échangeur pour chacun des débits volumiques d'entrée

    Args :
        par : dict, dictionnaire des paramètres de l'échangeur
        cond : dict, dictionnaire des conditions de l'étude (paramètres du fluide)
        list_Dv : list, liste des débits volumiques d'entrée dans l'échangeur
        fappx : float, 1/4 du facteur multiplicatif des pertes de charges sur la longueur des développement de l'écoulement dans les canaux

    Returns :
        list_PL : list, pertes de charges totales entrée/sortie de l'échangeur pour chacun des débits volumiques d'entrée
        list_tabl : list, état du système (pressions et débits dans l'échangeur) pour chacun des débits volumiques d'entrée
    """

    list_PL = []
    list_tabl = []

    for Dv in list_Dv:
        cond["Dv"] = Dv
        tabl, res, PrL, testings = PL_fsolve(par,cond, fappx = fappx)
        list_PL.append(res)
        list_tabl.append(tabl)

    return np.array(list_PL),list_tabl

def PL_fsolve_range_rd(par,cond,list_rd,fappx=0.25):
    """
    
    Args :
        par : dict, dictionnaire des paramètres de l'échangeur
        cond : dict, dictionnaire des conditions de l'étude (paramètres du fluide)
        list_rd : list, liste des rapports de diamètres riser/manifold
        fappx : float, 1/4 du facteur multiplicatif des pertes de charges sur la longueur des développement de l'écoulement dans les canaux
        
    Returns :
        list_PL : list, pertes de charges totales entrée/sortie de l'échangeur pour chacun des rapports de diamètres riser/manifold
        list_tabl : list, état du système (pressions et débits dans l'échangeur) pour chacun des rapports de diamètres riser/manifold
        """

    list_PL = []
    list_tabl = []
    D0_riser = par['D_riser']
    D0_man = par['D_man']
    r0 = D0_riser/D0_man
    def change_diameter(par, D, name):
        former = par['D_'+ name]
        par['D_'+ name] = D
        par['A_' + name] *= (D/former)**2
    
    for rd in list_rd:
        change_diameter(par, D0_man*np.sqrt(r0/rd), name='man')
        change_diameter(par, D0_riser*np.sqrt(rd/r0), name='riser')
        tabl, res, PL, residuals = PL_fsolve(par,cond,show=False, fappx=fappx)
        list_PL.append(res)
        list_tabl.append(tabl)
    return np.array(list_PL),list_tabl