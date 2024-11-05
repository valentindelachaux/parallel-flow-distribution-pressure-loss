import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import scipy.optimize as sc
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def logistique(x, depth, shift, steepness, c):
    y = c + depth / (1.0 + np.exp(steepness * (shift-x)))
    return y

### Classes ###
class flow_field:
    def __init__(self, X, par, cond):
        """
        Args:
            X (list): list of channel flow values
            par (dict): dictionary of heat exchanger parameters
            cond (condition): object containing the conditions of the fluid """

        self.X = np.array(X)
        for key in par.keys():
            setattr(self, key, par[key])
        self.cond = cond
        N=par["N"]
        ref = par["ref"]
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
        ep = par["roughness"]
        alpha = cond.alpha
        eta = cond.eta
        rho = cond.rho
        Qin_c = cond.Qin_c
        Qin_d = cond.Qin_d
        Qin = Qin_d - np.array([np.sum(self.X[i+1:]) for i in range(self.N)])
        if ref == 0 :
            Qout = Qin_c + np.array([np.sum(self.X[i:N]) for i in range(self.N)])
        else : 
            Qout = Qin_c + np.array([np.sum(self.X[:i+1]) for i in range(self.N)])

        uin = Qin/Ain
        ux = self.X/Ax
        uout = Qout/Aout
        Rein = fds.core.Reynolds(uin,D_in,rho,mu=eta)
        Rex = fds.core.Reynolds(ux,Dx,rho,mu=eta)
        Lex = 0.05*np.sqrt(4*Ax/np.pi)*Rex
        Reout = fds.core.Reynolds(uout,D_out,rho,mu=eta)
        fin = [fds.friction.friction_factor(Re = Rein[i],eD = ep/D_in) for i in range(self.N)]
        fx = [fds.friction.friction_factor(Re = Rex[i],eD=ep/Dx) for i in range(self.N)]
        fout = [fds.friction.friction_factor(Re = Reout[i],eD=ep/D_out) for i in range(self.N)]
        Kx_in = [Kxin(D_in,Dx,theta,Qin[i],self.X[i],i,c_Kxin, method) for i in range(self.N)]
        Ky_in = [0]+[Kyin(D_in,Dx,theta,Qin[i-1],self.X[i],i,c_Kyin, method=method) for i in range(1,self.N)]
        Kx_out = [Kxout(D_out,Dx,theta,Qout[i],self.X[i],i,c_Kxout, method=method) for i in range(self.N)]
        Ky_out = [Kyout(D_out,Dx,theta,Qout[i],self.X[i],i,c_Kyout, method=method) for i in range(self.N)]           

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


### Résolution pour un échangeur ###
def inlet_pressure(field):
    N=field.N
    ref = field.ref
    Dx = field.D_riser
    D_in = field.D_man
    Lx = field.L_riser
    Ly_list = field.Ly
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
            Pin[i] = Pin[i-1]+(rho/2)*(uin[i-1]**2-uin[i]**2 + fin[i]*(Ly_list[i-1]/D_in)*uin[i]**2 + Kyin[i]*uin[i]**2)
    else:
        for i in range(N-2,-1,-1):
            Pin[i] = Pin[i+1]+(rho/2)*(uin[i+1]**2-uin[i]**2 + fin[i]*(Ly_list[i]/D_in)*uin[i]**2 + Kyin[i]*uin[i]**2)
    
    return Pin

def f(field):
    N = field.N
    Dx = field.D_riser
    Lx = field.L_riser
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
    N = field.N
    D_out = field.D_man
    rho = field.cond.rho
    uout = field.uout
    Ky_out = field.Ky_out
    Ly_list = field.Ly
    fout = field.fout
    Pout = np.zeros(N)
    if field.ref == 0:
        for i in range(1,N):
            Pout[i] = Pout[i-1] + (rho/2)*(uout[i-1]**2 - uout[i]**2 + fout[i]*(Ly_list[i-1]/D_out)*uout[i]**2 + Ky_out[i]*uout[i]**2)
    else:
        for i in range(N-2,-1,-1):
            Pout[i] = Pout[i+1] + (rho/2)*(uout[i+1]**2 - uout[i]**2 + fout[i]*(Ly_list[i]/D_out)*uout[i]**2 + Ky_out[i]*uout[i]**2)

    return Pout

def phi(field):
    X = field.X
    alpha = field.cond.alpha
    Qin_d = field.cond.Qin_d

    return abs(np.sum(X)/alpha - Qin_d)*3600000

def PL_fsolve(par, cond, q_init=[], show_residuals=False):
    N = par["N"]
    if q_init == []:
        X0 = np.array([1]+[cond.Qin_d * cond.alpha/N]*N)
    else:
        X0 = np.array([1]+q_init)

    
    def fun(X):
        q_vect = X[1:]
        field = flow_field(q_vect, par, cond)
        equations = np.append(f(field) - g(field), phi(field))
        residuals.append(np.linalg.norm(equations))
        return equations
    
    residuals = []
    Xsol = sc.fsolve(fun, X0)

    if show_residuals:
        iterations = np.arange(1, len(residuals) + 1)
        plt.plot(iterations, residuals, marker='o')
        plt.xlabel('Nombre d\'itérations')
        plt.ylabel('Résidus')
        plt.title('Résidus en fonction du nombre d\'itérations')
        plt.grid(True)
        plt.show()

    q_sol = Xsol[1:]
    field = flow_field(q_sol, par, cond)
    tabl = pd.DataFrame({"Pin":inlet_pressure(field), "Pout":f(field), "qx":q_sol*3600000})
    tabl = tabl[::-1].reset_index(drop=True)

    return tabl, fun(Xsol)


### Résolution mettre les échangeurs en série ###
def range_cond_solve(par, list_Qmax, list_proportion = np.arange(0.1, 1.1, 0.1) , list_alpha= np.arange(0.1, 1.1, 0.1), eta=1e-3, rho=1000):
    """Solve the pressure losses for a range of conditions
    Args:
        par (dict): dictionary of heat exchanger parameters
        list_Qmax (list): list of flow rates
        list_proportion (list): list of proportion of flow rate in the collector or the distributor
        list_alpha (list): list of alpha values
        eta (float): fluid viscosity
        rho (float): fluid density
        
    Returns:
        df_results (DataFrame): dataframe containing the results of the pressure losses (abaque pour la méthode de résolution par fonctions de transfert)"""
    
    N = par["N"]
    ref = par["ref"]
    df_cond = pd.DataFrame()
    for Qmax in list_Qmax:
        list_Qin_d = Qmax*list_proportion
        list_Qin_c = Qmax - list_Qin_d
        df_Q = pd.DataFrame({'Qmax':Qmax, 'Qin_d':list_Qin_d, 'Qin_c':list_Qin_c})
        df_Q_alpha = df_Q.merge(pd.DataFrame({'alpha' : list_alpha}), how='cross')
        df_cond = pd.concat([df_cond, df_Q_alpha], ignore_index=True)
    df_cond['eta'] = eta
    df_cond['rho'] = rho

    df_results = pd.DataFrame()
    for i in range(len(df_cond)):
        cond = condition(df_cond.loc[i])
        tabl, residuals = PL_fsolve(par, cond)
        df_DP = pd.DataFrame({'P_coll_inlet':tabl.iloc[N-1 - ref]['Pout'], 'P_distrib_inlet':tabl.iloc[N-1 - ref]['Pin'], 'P_distrib_outlet':tabl.iloc[ref]['Pout']}, index=[i])
        df_results = pd.concat([df_results, df_DP], ignore_index=True)
    
    df_results = df_cond.join(df_results)
    return df_results

# def transfer_func(df_testings, type='polynomial', deg_a=2, deg_q =2, initial_guess_logistic=[-3, 0.4, 1, 1]):
#     """
#     Créer les fonctions de transfert des pertes de charges totales en fonction de Qin_d et alpha, à Qmax donné (df_testings est à Qmax donné)

#     Args : 
#         df_testings : dataframe contenant les tests de pertes de charges
#         type : str, 'polynomial' ou 'logistic'
#         degree : int, degré du polynôme 
        
#     Returns :
#         DPin : fonction de transfert des pertes de charges régulières manifold inlet
#         DPout : fonction de transfert des pertes de charges régulières manifold outlet
#         DPx : fonction de transfert des pertes de charges régulières canaux
#     """
#     # df_testings['DPd'] = df_testings['Pin_d'] - df_testings['Pout_d']

#     # trouver les coefficients tels que DP = a*alpha^2 + b*alpha + c , à Qin donné
#     if type == 'polynomial':
#         n = deg_a+1
#     elif type == 'logistic':
#         n = 4
#     columns = ['Qin_d'] + [f'D_{i}' for i in range(n)] + [f'C_{i}' for i in range(deg_a+1)] + [f'DC_{i}' for i in range(deg_a+1)]
#     df_coefficients = pd.DataFrame(columns=columns)
#     Qin_list = df_testings['Qin_d'].unique()
#     for Qin in Qin_list:
#         mask = df_testings['Qin_d'] == Qin
#         alpha = np.array(df_testings[mask]['alpha'])  
#         DPd = np.array(df_testings[mask]['DPd']) 
#         # DPc = np.array(df_testings[mask]['Pin_c'])
#         DPc = np.array(df_testings[mask]['DPc'])
#         # DPdc = np.array(df_testings[mask]['Pout_d'])
#         DPdc = np.array(df_testings[mask]['DPdc'])
#         coefficients_c = np.polyfit(alpha, DPc, deg_a)
#         coefficients_dc = np.polyfit(alpha, DPdc, deg_a)
#         if type == 'polynomial':
#             coefficients_d = np.polyfit(alpha, DPd, deg_a)
#         else :
#             coefficients_d, _ = curve_fit(logistique, alpha, DPd, p0=initial_guess_logistic, maxfev=1000000)

#         df_coefficients.loc[len(df_coefficients)] = [Qin] + list(coefficients_d) + list(coefficients_c) + list(coefficients_dc)

#     # trouver la matrice tels que les coefficients (a, b, c) = M.(Qin^2, Qin, 1)
#     def M(Qin_list, df_coefficient):
#         matrix = pd.DataFrame(columns=[i for i in range(deg_q+1)])
#         for column in df_coefficient.columns:
#             COEFF = np.polyfit(Qin_list, list(df_coefficient[column]), deg_q)
#             matrix.loc[len(matrix)] = COEFF
#         return matrix
    
#     coefficients_d = df_coefficients[[f'D_{i}' for i in range(n)]]
#     coefficients_c = df_coefficients[[f'C_{i}' for i in range(deg_a+1)]]
#     coefficients_dc = df_coefficients[[f'DC_{i}' for i in range(deg_a+1)]]

#     MATRIX_d = M(Qin_list, coefficients_d)
#     MATRIX_c = M(Qin_list, coefficients_c)
#     MATRIX_dc = M(Qin_list, coefficients_dc)

#     # la fonction DP = f(Qin, alpha) = [M.(Qin^2, Qin, 1)].(alpha^2, alpha, 1)
#     def DPd(Qin, alpha):
#         coeffs = np.matmul(MATRIX_d, np.array([Qin**(deg_q-i) for i in range(deg_q+1)]))
#         if type == 'polynomial':
#             PL = np.matmul(coeffs, np.array([alpha**(deg_a-i) for i in range(deg_a+1)]))
#         else:
#             PL = logistique(alpha, *coeffs)
#         return PL

#     def DPc(Qin, alpha):
#         coeffs = np.matmul(MATRIX_c, np.array([Qin**(deg_q-i) for i in range(deg_q+1)]))
#         return np.matmul(coeffs, np.array([alpha**(deg_a-i) for i in range(deg_a+1)]))

#     def DPdc(Qin, alpha):
#         coeffs = np.matmul(MATRIX_dc, np.array([Qin**(deg_q-i) for i in range(deg_q+1)]))
#         return np.matmul(coeffs, np.array([alpha**(deg_a-i) for i in range(deg_a+1)]))

#     return DPd, DPc, DPdc

def transfer_func(df_testings, deg_a=2, deg_q=2):
    """
    Creates transfer functions for total pressure drop as a function of Qin_d and alpha at a given Qmax (df_testings should be filtered for a specific Qmax).

    Args:
        df_testings: DataFrame containing pressure drop test results.
        deg_a: int, degree of the polynomial in terms of alpha.
        deg_q: int, degree of the polynomial in terms of Qin.

    Returns:
        DPd: function for transfer of regular manifold inlet pressure drops.
        DPc: function for transfer of regular manifold outlet pressure drops.
        DPdc: function for transfer of regular channel pressure drops.
    """
    # Identify unique flow rates and create a DataFrame to store polynomial coefficients
    n = deg_a + 1
    columns = ['Qin_d'] + [f'D_{i}' for i in range(n)] + [f'C_{i}' for i in range(n)] + [f'DC_{i}' for i in range(n)]
    df_coefficients = pd.DataFrame(columns=columns)

    # Process each unique Qin value to fit polynomial coefficients for DPc, DPd, and DPdc
    Qin_list = df_testings['Qin_d'].unique()
    for Qin in Qin_list:
        mask = df_testings['Qin_d'] == Qin
        alpha = np.array(df_testings[mask]['alpha'])
        DPd = np.array(df_testings[mask]['DPd'])
        DPc = np.array(df_testings[mask]['DPc'])
        DPdc = np.array(df_testings[mask]['DPdc'])

        coefficients_d = np.polyfit(alpha, DPd, deg_a)
        coefficients_c = np.polyfit(alpha, DPc, deg_a)
        coefficients_dc = np.polyfit(alpha, DPdc, deg_a)

        df_coefficients.loc[len(df_coefficients)] = [Qin] + list(coefficients_d) + list(coefficients_c) + list(coefficients_dc)

    # Define a function to calculate the transformation matrix for polynomial coefficients with respect to Qin
    def create_transformation_matrix(Qin_list, coefficients_df):
        matrix = pd.DataFrame(columns=[i for i in range(deg_q + 1)])
        for column in coefficients_df.columns:
            poly_coeffs = np.polyfit(Qin_list, list(coefficients_df[column]), deg_q)
            matrix.loc[len(matrix)] = poly_coeffs
        return matrix

    # Generate transformation matrices for each pressure drop type
    MATRIX_d = create_transformation_matrix(Qin_list, df_coefficients[[f'D_{i}' for i in range(n)]])
    MATRIX_c = create_transformation_matrix(Qin_list, df_coefficients[[f'C_{i}' for i in range(n)]])
    MATRIX_dc = create_transformation_matrix(Qin_list, df_coefficients[[f'DC_{i}' for i in range(n)]])

    # Define the transfer functions for DPd, DPc, and DPdc
    def DPd(Qin, alpha):
        coeffs = np.matmul(MATRIX_d, np.array([Qin ** (deg_q - i) for i in range(deg_q + 1)]))
        return np.matmul(coeffs, np.array([alpha ** (deg_a - i) for i in range(deg_a + 1)]))

    def DPc(Qin, alpha):
        coeffs = np.matmul(MATRIX_c, np.array([Qin ** (deg_q - i) for i in range(deg_q + 1)]))
        return np.matmul(coeffs, np.array([alpha ** (deg_a - i) for i in range(deg_a + 1)]))

    def DPdc(Qin, alpha):
        coeffs = np.matmul(MATRIX_dc, np.array([Qin ** (deg_q - i) for i in range(deg_q + 1)]))
        return np.matmul(coeffs, np.array([alpha ** (deg_a - i) for i in range(deg_a + 1)]))

    return DPd, DPc, DPdc

def PL_fsolve_MPE(N_MPE, Qmax, DPd, DPc, DPdc):
    """
    Résoud les pertes de charges et les débits d'une ligne de panneaux/MPE/groupe de canaux caractérisés par ses fonctions de transfert DPx

    Args :
        N_MPE : int, nombre de groupes de canaux
        Qmax : float, débit total
        DPd : fonction de transfert des pertes de charges manifold inlet
        DPc : fonction de transfert des pertes de charges manifold outlet
        DPdc : fonction de transfert des pertes de charges canaux

    Returns :
        tabl : DataFrame, tableau des pertes de charges et débits
        FUN(Xsol) : array, résidus de la solution
    """

    #X[0] est celui le plus éloigné de l'entrée

    def PHI(X):
        return (np.sum(X) - Qmax)*3600000
    
    def G(X):
        Q_c = np.array([Qmax - sum(X[i+1:]) for i in range(N_MPE)])
        Pout_list = [sum([DPc(Q_c[j],X[j]/Q_c[j]) for j in range(i)]) for i in range(N_MPE)]
        return np.array(Pout_list)
    
    def IN_P(X):
        Q_c = np.array([Qmax - sum(X[i+1:]) for i in range(N_MPE)])
        Pin_list = [DPdc(Q_c[0],X[0]/Q_c[0])+ sum([DPd(Q_c[j],X[j]/Q_c[j]) for j in range(1,i+1)]) for i in range(N_MPE)]
        return np.array(Pin_list)       

    def F(X):
        Q_c = np.array([Qmax - sum(X[i+1:]) for i in range(N_MPE)])
        P_c = IN_P(X)
        Pout_list = [P_c[i] - DPd(Q_c[i],X[i]/Q_c[i]) - DPdc(Q_c[i],X[i]/Q_c[i]) for i in range(N_MPE)]
        return np.array(Pout_list)

    def FUN(X):
        q_vect = X[1:]
        equations = np.append(F(q_vect) - G(q_vect), PHI(q_vect))
        return equations
    
    X0 = np.array([0]+[Qmax/N_MPE]*N_MPE)
    Xsol = sc.fsolve(FUN, X0)

    q_sol = Xsol[1:]
    tabl = pd.DataFrame({"Pin":IN_P(q_sol), "Pout":F(q_sol), "qx":q_sol*3600000})
    tabl = tabl[::-1].reset_index(drop=True)

    return(tabl, FUN(Xsol))


### work in progress ###

def Triple_ax(Q):
    rho = 1000
    eta = 1e-3
    ep = 1e-3

    Dx = 12e-3
    Lx = 24.97
    R = 35e-3
    Ax = np.pi*(Dx/2)**2
    ux = Q/Ax

    # Pertes régulières
    Rex = fds.core.Reynolds(ux,Dx,rho,mu=eta)
    fx = fds.friction.friction_factor(Rex,eD=ep/Dx)
    ax_RPL = fx*(Lx/Dx)

    # Pertes singulières
    K_R = fds.fittings.bend_rounded(Dx, 180, rc=R, method='Crane')
    K_r = fds.fittings.bend_rounded(Dx, 180, rc=R/2, method='Crane')
    K_90 = fds.fittings.bend_rounded(Dx, 90, rc=R, method='Crane')
    ax_SPL = (20*K_R + 2*K_r + 4*K_90)

    return ax_RPL+ax_SPL
