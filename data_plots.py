import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import model_fsolve as modf

### Pressure losses plots 
def PL_percent(PL):
    PL_per = pd.DataFrame((list(zip(PL["Total PL"], 100*PL["SPL entrance"]/PL["Total PL"], 100*PL["RPL manifold"]/PL["Total PL"], 100*PL["RPL riser"]/PL["Total PL"], 100*PL["SPL tee"]/PL["Total PL"]))), columns = ["Total PL", "SPL entrance", "RPL manifold", "RPL riser", "SPL tee"])
    PL_per.reset_index().plot(x='index', y= ["SPL entrance", "RPL manifold", "RPL riser", "SPL tee"], style='o', xlabel='N° riser', ylabel='% head loss contribution')

def PL_hist(PL):
    PL.plot.bar(y=['SPL entrance', 'RPL manifold', 'RPL riser', 'SPL tee', 'Pressure recovery'], stacked = True, xlabel='N° riser', ylabel='Pressure loss')

def PL_dv(par, condi0, list_Vdot, lab=None):
    list_Dv = list_Vdot/3600000 # m3/s
    list_PL,list_tabl,list_mn,list_std = modf.PL_fsolve_range(par,condi0,list_Dv)
    plt.plot(list_Vdot, list_PL/1000,'-o', label=lab)
    plt.xlabel('Flow rate [L/h]')
    plt.ylabel('Pressure loss [kPa]')
    plt.legend()
    return(list_PL, list_tabl, list_mn, list_std)

def PL_ratio(par, condi0, list_Vdot, list_rd):
    for rd in list_rd:
        modf.change_diameter(par, par['D_riser']/rd, name='man')
        PL_dv(par,condi0,list_Vdot, lab=f'Rd = {rd}')

### Flow distribution plots

def beta_riser(tabl, axis=None, lab=None):
    list_qx = tabl['qx'][:]
    N = len(list_qx)
    list_qx.reset_index(drop=True,inplace=True)

    list_beta = list_qx/list_qx.sum()

    plt.plot([i for i in range(N)], list_beta, '-o', label=lab)
    plt.axis(axis)
    plt.xlabel('N° riser')
    plt.ylabel('Beta_i')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2),ncol=2)

def beta_vdot(par,condi0,list_Vdot):
    list_Dv = list_Vdot/3600000 # m3/s
    list_PL,list_tabl,list_mn,list_std = modf.PL_fsolve_range(par,condi0,list_Dv)
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