import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import model_fsolve as modf
import from_excel as fe

### Pressure losses plots 
def PL_percent(PL):
    PL_per = pd.DataFrame((list(zip(PL["Total PL"], 100*PL["SPL entrance"]/PL["Total PL"], 100*PL["RPL manifold"]/PL["Total PL"], 100*PL["RPL riser"]/PL["Total PL"], 100*PL["SPL tee"]/PL["Total PL"]))), columns = ["Total PL", "SPL entrance", "RPL manifold", "RPL riser", "SPL tee"])
    PL_per.plot(y= ["SPL entrance", "RPL manifold", "RPL riser", "SPL tee"], xlabel='N° riser', ylabel='% head loss contribution')

def PL_hist(PL):
    PL.plot.bar(y=['SPL entrance', 'RPL manifold', 'RPL riser', 'SPL tee', 'Pressure recovery'], stacked = True, xlabel='N° riser', ylabel='Pressure loss [Pa]', xticks= np.linspace(0, len(PL), 20))

def PL_dv(list_Vdot, list_PL, lab=None, unit = None):
    if unit == 'L/min':
        plt.plot(list_Vdot/60, list_PL/1000,'-o', label=lab)
        plt.xlabel('Flow rate [L/min]')
    else :
        plt.plot(list_Vdot, list_PL/1000,'-o', label=lab)
        plt.xlabel('Flow rate [L/h]')
    plt.ylabel('Pressure loss [kPa]')
    plt.legend()

def PL_ratio(par, cond, list_Vdot, list_rd, unit=None):
    D0_riser = par['D_riser']
    D0_man = par['D_man']
    r0 = D0_riser/D0_man
    for rd in list_rd:
        modf.change_diameter(par, D0_man*np.sqrt(r0/rd), name='man')
        modf.change_diameter(par, D0_riser*np.sqrt(rd/r0), name='riser')
        list_PL, list_tabl = modf.PL_fsolve_range(par, cond, np.array(list_Vdot)/3600000)
        PL_dv(list_Vdot, list_PL, lab=f'Rd = {rd}', unit=unit)

### Flow distribution plots

def beta_riser(tabl, axis=None, lab=None, dimensionless=True, rho=None):
    list_qx = tabl['qx'][:]
    N = len(list_qx)
    list_qx.reset_index(drop=True,inplace=True)
    if dimensionless:
        list_beta = list_qx/list_qx.mean()
        plt.plot([i for i in range(N)], list_beta, label=lab)
        plt.axis(axis)
        plt.ylim(ymin=0,ymax=max(list_beta)+0.2)
        plt.xlabel('N° riser')
        plt.ylabel('q_x / q_moy')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),ncol=2)
    else :
        plt.plot([i for i in range(N)], list_qx*rho/3600000, label=lab)
        plt.axis(axis)
        plt.ylim(ymin=0)
        plt.xlabel('N° riser')
        plt.ylabel('Mass flow rate [kg/s]')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),ncol=2)    

def beta_rd(list_rd, list_tabl):
    for i in range(len(list_tabl)):
        beta_riser(list_tabl[i], lab=f'rd = {list_rd[i]} L/h')

def beta_vdot(list_Vdot, list_tabl):
    for i in range(len(list_tabl)):
        beta_riser(list_tabl[i], lab=f'flow rate = {list_Vdot[i]: .1f} L/h')

def beta_lriser(par,condi0,list_lriser):
    list_tabl = []
    for lriser in list_lriser: 
        par['L_riser'] = lriser
        tabl, res, PL = modf.PL_fsolve(par,condi0,print=False)
        list_tabl.append(tabl)
    for i in range(len(list_tabl)):
        beta_riser(list_tabl[i], lab=f'Riser length = {list_lriser[i]: .1f} m')


def u_rd(par, condi0, list_rd, name = 'uin',lab=None):
    for rd in list_rd:
        modf.change_diameter(par, par['D_riser']/rd, name='man')
        tabl, res, PL, u, K = modf.PL_fsolve(par,condi0,print=False)
        plt.plot(u[name],'-o', label=f'rd = {rd}')
    plt.xlabel('riser')
    plt.ylabel(f'{name} [m/s]')
    plt.legend()
    plt.show()

def K_rd(par, condi0, list_rd, name = 'Kx_in',lab=None):
    for rd in list_rd:
        modf.change_diameter(par, par['D_riser']/rd, name='man')
        tabl, res, PL,u,K = modf.PL_fsolve(par,condi0,print=False)
        plt.plot(K[name],'-o', label=f'rd = {rd}')
    plt.xlabel('riser')
    plt.ylabel(f'{name}')
    plt.legend()
    plt.show()


### SOLVE AND PLOT

def solve_plot(path, file_name, hist_pl = True, perc_pl = True, pl_dv = True, beta_dv = True, list_Vdot = np.array([50, 200, 350, 500]), series = False):
    
    hx, par, condi0 = fe.initialize(path, file_name)

    tabl, res, PL, testings = modf.PL_fsolve(par,condi0, show=False, series=series)
    if hist_pl :
        PL_hist(PL)
        plt.show()
    if perc_pl :
        PL_percent(PL)
        plt.show()

    if pl_dv or beta_dv :
        list_PL, list_tabl= modf.PL_fsolve_range(par, condi0, np.array(list_Vdot)/3600000, 20)     

        if pl_dv :
            PL_dv(list_Vdot, list_PL)
            plt.show()
        if beta_dv :
            beta_vdot(list_Vdot, list_tabl)
            plt.show()

    PL_ratio(par, condi0, list_Vdot, [0.25, 0.5, 1.], unit=None)

### Abaque K 

def K_abaque(df_testings, K_name):
    unique_QF = df_testings['QF'].unique()
    unique_QF_out = df_testings['QF_out'].unique()

    fig, axes = plt.subplots(nrows=len(unique_QF), ncols=len(unique_QF_out), figsize=(15, 10))
    K_max = df_testings[K_name].max()
    K_min = df_testings[K_name].min()

    for i, QF in enumerate(unique_QF):
        for j, QF_out in enumerate(unique_QF_out):
            filtered_data = df_testings[(df_testings['QF'] == QF) & (df_testings['QF_out'] == QF_out)]
            ax = axes[i, j]
            ax.plot(filtered_data['alpha'], filtered_data[K_name], label = 'Original model')
            ax.plot(filtered_data['alpha'], filtered_data[K_name+'_test'], label = 'Least squares model')
            # if K_max*K_min < 0 :
            #     ax.set(ylim=(K_min, K_max))
            # elif K_min <=0 :
            #     ax.set(ylim=(K_min, 0))
            # else :
            #     ax.set(ylim=(0, K_max))
            ax.legend()
            ax.set_title(f'Vdot: {QF*3600000:.0f} L/h, Vdot_out: {QF_out*3600000:.0f} L/h')
            ax.set_xlabel('Alpha')
            ax.set_ylabel(K_name)

    plt.tight_layout()
    plt.show()