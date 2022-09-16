import math
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids as fds
import scipy.optimize as sc

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as mtick
from matplotlib.cm import get_cmap

import model2 as mod
import model_fsolve as modf

import fluids as fds

def preproc(par,param_name,test_value):
    if param_name == "h_riser":
        par["h_riser"] = test_value
        par["Dx"] = 2*(par["h_riser"]*par["l_riser"])/(par["h_riser"]+par["l_riser"])
    elif param_name == "N_riser":
        par["N"] = test_value
        par["l_riser"] = (par["tot_width"]-par["inter_riser"]*(par["N"]-1))/par["N"]

        par["N_per_EP"] = int(par["N"]/par["EP"])

        petit = (par["N_per_EP"]-1)*[par["inter_riser"]]
        long = [0.013]
        par["Ly"] = (par["EP"]-1)*(petit+long)+petit # m
    else:
        pass

def modf_parametric_flow_rates(par,list_Q):
    list_PL = []
    list_tabl = []
    list_mn = []
    list_std = []

    for Q in list_Q:
        print(Q)
        par["QF"] = Q
        # Speed and Reynolds at inlet manifold
        par["U"] = par["QF"]/par["Ain"]
        par["Reman"] = par["U"]*(par["rho"]*par["Din"])/par["eta"]

        tabl,res = modf.PL_fsolve(par,par["sch"],False)
        list_PL.append(res)
        list_tabl.append(tabl)

        list_mn.append(tabl['qx'].mean()) # fow rate qx is in L/h
        list_std.append(tabl['qx'].std())
    
    return list_PL,list_tabl,list_mn,list_std


# Calcul des contributions des différents éléments -> procédure modifiant les éléments de list_tabl

def repartition_PL(par,list_PL,list_tabl,list_Q):

    rho = par['rho']

    for q in range(len(list_Q)):
        PLq = list_PL[q]

        lin_in = []
        lin_x = []
        lin_out = []
        lin_in_cum = []
        lin_out_cum = []
        sing = []

        tab = list_tabl[q]

        Qin = []
        Qout = []
        
        for i in range(par["N"]):
            Qin.append(sum([tab['qx'][j] for j in range(0,i+1)]))
            Qout.append(sum([tab['qx'][j] for j in range(i,par["N"])]))

        tab['Qin'] = Qin
        tab['Qout'] = Qout

        for i in range(par["N"]):
            Qin_i = tab['Qin'][i]
            Qout_i = tab['Qout'][i]
            qx_i = tab['qx'][i]

            uin_i = (Qin_i/3600000)/par['Ain']
            ux_i = (qx_i/3600000)/par['Ax']
            uout_i = (Qout_i/3600000)/par['Aout']

            Rein_i = fds.core.Reynolds(uin_i,par['Din'],par['rho'],mu=par['eta'])
            Rex_i = fds.core.Reynolds(ux_i,par['Dx'],par['rho'],mu=par['eta'])
            Reout_i = fds.core.Reynolds(uout_i,par['Dout'],par['rho'],mu=par['eta'])
            fin_i = fds.friction.friction_factor(Re = Rein_i)
            fx_i = fds.friction.friction_factor(Re = Rex_i)
            fout_i = fds.friction.friction_factor(Re = Reout_i)

            Ly_arr = np.array(par['Ly'])
            Ly_av = np.average(Ly_arr)


            ain_i = fin_i*(Ly_av/par['Din'])
            ax_i = fx_i*(par['Lx']/par['Dx'])
            aout_i = fout_i*(Ly_av/par['Dout'])

            lin_in.append((rho/2)*ain_i*uin_i**2)
            lin_x.append((rho/2)*ax_i*ux_i**2)
            lin_out.append((rho/2)*aout_i*uout_i**2)

        tab['lin_in'] = lin_in
        tab['lin_x'] = lin_x
        tab['lin_out'] = lin_out

        for i in range(par["N"]):
            lin_in_cum.append(sum([lin_in[j] for j in range(i,par["N"])]))
            lin_out_cum.append(sum([lin_out[j] for j in range(0,i+1)]))

        tab['lin_in_cum'] = lin_in_cum
        tab['lin_out_cum'] = lin_out_cum

        for i in range(par["N"]):
            sing.append(PLq-tab['lin_in_cum'][i]-tab['lin_x'][i]-tab['lin_out_cum'][i])
        
        tab['sing'] = sing

# Plot la répartition des pertes de charge entre régulières, manifold et riser, singulières

def plt_repartition_PL(par,list_Q_L,list_PL,list_tabl,flow_rate,disp):
    
    Q = flow_rate
    list_Q_L_l = list(list_Q_L)
    q = list_Q_L_l.index(Q)
    print(Q)

    if disp == "too_much":
        N_disp = (par["N"]//10)+1
        x = np.array(range(0,N_disp))
        x = 10*x
    else:
        N_disp = par["N"]
        x = np.array(range(0,N_disp))

    print(x)
    print(len(x))

    # Stacked bars chart

    labels = [str(x[i]) for i in range(len(x))]

    tab = list_tabl[q]

    lin_in_cum = []
    lin_x = []
    lin_out_cum = []
    sing = []

    for i in range(len(x)):
        lin_in_cum.append(tab['lin_in_cum'][x[i]])
        lin_x.append(tab['lin_x'][x[i]])
        lin_out_cum.append(tab['lin_out_cum'][x[i]])
        sing.append(tab['sing'][x[i]])

    lin_in_cum = np.array(lin_in_cum)
    lin_x = np.array(lin_x)
    lin_out_cum = np.array(lin_out_cum)
    sing = np.array(sing)

    width = 0.35       # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, lin_in_cum, width, label='Inlet header linear PL')
    ax.bar(labels, lin_x, width, bottom=lin_in_cum,label='Riser linear PL')
    ax.bar(labels, lin_out_cum, width, bottom=lin_in_cum+lin_x,label='Outlet header linear PL')
    ax.bar(labels, sing, width, bottom=lin_in_cum+lin_x+lin_out_cum,label='Singular PL')

    ax.plot(labels,np.array(N_disp*[list_PL[q]]))

    ax.set_ylabel('PL (Pa)')
    # ax.set_title('SPRING')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    print(Q)
    plt.show()


# Calcul des pertes de charge régulière de l'écoulement dans un tube

def lin_PL():

    fr = 0.0035 #kg/s
    rho_sw = 985
    mu_sw = 1*1E-3 # viscosité dynamique de l'eau
    D_sw = 0.008
    A_sw = math.pi*(D_sw/2)**2
    u_sw = (fr/rho_sw)/A_sw # eau à rho

    Re_sw = fds.core.Reynolds(u_sw,D_sw,rho_sw,mu=mu_sw)
    print(Re_sw)
    L_sw = 0.100

    f_weib = fds.friction.friction_factor(Re = Re_sw)
    K_sw = f_weib*(L_sw/D_sw)

    PL_sw = (rho_sw/2)*K_sw*u_sw**2

    print(PL_sw)


def sing_PL(par,d_epdm,d_red,d_man,L_epdm,angle_bm_epdm,angle_bm_man,list_Q_L,list_PL):

    list_Q = list_Q_L/3600000

    list_Re = []

    Kc_list_R = []
    Kc_list_H = []
    PL_c_R = []
    PL_c_H = []

    Kdm_list_R = []
    Kdm_list_H = []
    PL_dm_R = []
    PL_dm_H = []

    Kcm_list_R = []
    Kcm_list_H = []
    PL_cm_R = []
    PL_cm_H = []

    Kd_list_R = []
    Kd_list_H = []
    PL_d_R = []
    PL_d_H = []

    list_Kbm_man = []
    PL_bm_man = []

    list_Kbm_epdm = []
    PL_bm_epdm = []

    PL_epdm = []

    for Q in list_Q:
        u_epdm = Q/(math.pi*(d_epdm/2)**2)
        u_red = Q/(math.pi*(d_red/2)**2)
        u_man = Q/(math.pi*(d_man/2)**2)

        # Reynolds dans les liaisons EPDM, dans la réduction du QDF, dans les manifolds
        Re_epdm = fds.core.Reynolds(u_epdm,d_epdm,par["rho"],mu=par["eta"])
        list_Re.append(Re_epdm)
        Re_red = fds.core.Reynolds(u_red,d_red,par["rho"],mu=par["eta"])
        Re_man = fds.core.Reynolds(u_man,d_man,par["rho"],mu=par["eta"])

        # Contraction sharp EPDM-reduction
        Kc_H = fds.fittings.contraction_sharp(d_epdm,d_red,Re=Re_epdm,roughness=par["rough"],method='Hooper')
        Kc_R = fds.fittings.contraction_sharp(d_epdm,d_red,Re=Re_epdm,roughness=par["rough"],method='Rennels')
        Kc_list_H.append(Kc_H)
        Kc_list_R.append(Kc_R)
        PL_c_H.append((par["rho"]/2)*Kc_H*u_epdm**2)
        PL_c_R.append((par["rho"]/2)*Kc_R*u_epdm**2)

        # Diffuser sharp reduction-man
        Kdm_H = fds.fittings.diffuser_sharp(d_red,d_man,Re=Re_red,roughness=par["rough"],method='Hooper')
        Kdm_R = fds.fittings.diffuser_sharp(d_red,d_man,Re=Re_red,roughness=par["rough"],method='Rennels')
        Kdm_list_H.append(Kdm_H)
        Kdm_list_R.append(Kdm_R)
        PL_dm_H.append((par["rho"]/2)*Kdm_H*u_red**2)
        PL_dm_R.append((par["rho"]/2)*Kdm_R*u_red**2)

        # Contraction sharp manifold-reduction

        Kcm_H = fds.fittings.contraction_sharp(d_man,d_red,Re=Re_man,roughness=par["rough"],method='Hooper')
        Kcm_R = fds.fittings.contraction_sharp(d_man,d_red,Re=Re_man,roughness=par["rough"],method='Rennels')
        Kcm_list_H.append(Kcm_H)
        Kcm_list_R.append(Kcm_R)
        PL_cm_H.append((par["rho"]/2)*Kcm_H*u_man**2)
        PL_cm_R.append((par["rho"]/2)*Kcm_R*u_man**2)

        # Diffuser sharp reduction-EPDM
        Kd_H = fds.fittings.diffuser_sharp(d_red,d_epdm,Re=Re_red,roughness=par["rough"],method='Hooper')
        Kd_R = fds.fittings.diffuser_sharp(d_red,d_epdm,Re=Re_red,roughness=par["rough"],method='Rennels')
        Kd_list_H.append(Kd_H)
        Kd_list_R.append(Kd_R)
        PL_d_H.append((par["rho"]/2)*Kd_H*u_red**2)
        PL_d_R.append((par["rho"]/2)*Kd_R*u_red**2)

        # Bend miter in the EPDM
        Kbm_epdm = fds.fittings.bend_miter(angle_bm_epdm, Di=d_epdm, Re=fds.core.Reynolds(u_epdm,d_epdm,par["rho"],mu=par["eta"]),roughness=par["rough"], L_unimpeded=d_epdm, method='Rennels')
        list_Kbm_epdm.append(Kbm_epdm)
        PL_bm_epdm.append((par["rho"]/2)*Kbm_epdm*u_epdm**2)

        # Bend miter in the manifold
        Kbm = fds.fittings.bend_miter(angle_bm_man, Di=d_man, Re=Re_man, L_unimpeded=2*d_man, method='Rennels')
        list_Kbm_man.append(Kbm)
        PL_bm_man.append((par["rho"]/2)*Kbm*u_man**2)

        # Linear pressure losses

        f_epdm = fds.friction.friction_factor(Re = Re_epdm,eD=par["rough"]/d_epdm)
        K_lin_epdm = f_epdm*(L_epdm/d_epdm)
        PL_epdm.append((par["rho"]/2)*K_lin_epdm*u_epdm**2)

    # Total pressure losses

    exp_PL_tot = [(3.3*1E-5*Q**2+0.002363466*Q)*1000 for Q in list_Q_L]

    PL_tot = np.array(PL_c_H) + np.array(PL_dm_H) + np.array(PL_d_H) + np.array(PL_cm_H) + 2*np.array(PL_bm_epdm) + 2*np.array(PL_bm_man) + np.array(list_PL)
    PL_tot_lin = PL_tot + np.array(PL_epdm)

    print(list_Q_L)
    print(exp_PL_tot)
    print(PL_tot)

    plt.plot(list_Q_L,list_Re,label='Reynolds in EPDM')
    plt.legend()
    plt.show()

    plt.plot(list_Q_L,PL_c_R,label='Rennels')
    plt.plot(list_Q_L,PL_c_H,label='Hooper')
    plt.legend()
    plt.show()

    plt.plot(list_Q_L,PL_d_R,label='Rennels')
    plt.plot(list_Q_L,PL_d_H,label='Hooper')
    plt.legend()
    plt.show()

    plt.plot(list_Q_L,PL_bm_epdm,label='Rennels bm_epdm')
    plt.legend()
    plt.show()

    plt.plot(list_Q_L,PL_bm_man,label='Rennels bm_man')
    plt.legend()
    plt.show()

    plt.plot(list_Q_L,PL_epdm,label='Linear pressure losses')
    plt.legend()
    plt.show()

    plt.plot(list_Q_L,PL_tot,label='Estimation of SPRING PL')
    plt.plot(list_Q_L,PL_tot_lin,label='Estimation + 2 m of EPDM')
    plt.plot(list_Q_L,exp_PL_tot,label='Measured SPRING PL')
    plt.legend()
    plt.xlabel('Q (L/h)')
    plt.ylabel('PL (Pa)')
    plt.grid()
    plt.show()

    # cont_diff = np.array(PL_c_R) + np.array(PL_dm_R) + np.array(PL_d_R) + np.array(PL_cm_R)
    cont_diff = np.array(PL_c_H) + np.array(PL_dm_H) + np.array(PL_d_H) + np.array(PL_cm_H)
    bends = 2*np.array(PL_bm_epdm) + 2*np.array(PL_bm_man)
    harp = np.array(list_PL)
    reg = np.array(PL_epdm)

    return PL_tot,PL_tot_lin,exp_PL_tot,cont_diff,bends,harp,reg

# Plot pressure losses of harp

def plot_PL_harp(par,list_Q_L,list_PL,a,b,list_SW,end = None):
    # Pressure losses

    # Reference 

    if end !=None:
        list_Q_L = list_Q_L[0:end]
        list_PL = list_PL[0:end]

    PL_measured = (a*list_Q_L**2 + b*list_Q_L)

    plt.scatter(list_Q_L,PL_measured,label='Datasheet')

    plt.scatter(list_Q_L,list_SW,label = "SW Flow Simulation")

    # Plot pressure losses

    plt.plot(np.array(list_Q_L),np.array(list_PL),label='model')

    plt.plot(np.array(list_Q_L),np.array(list_PL)*0.8,'--',label='model-20%')

    plt.plot(np.array(list_Q_L),np.array(list_PL)*1.2,'--',label='model+20%')

    plt.xlabel('Q (L/h)')
    plt.ylabel('PL (Pa)')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.legend()

# Plot répartition des débits par canal (en valeur absolue)

def plot_abs_flow_rates_harp(par,list_Q_L,list_tabl,list_mn):

    list_Q_L_round = [round(num, 0) for num in list_Q_L]
    risers = np.linspace(0,par["N"]-1,par["N"])

    for i in range(len(list_tabl)):
        plt.plot(risers,np.array(list_tabl[i]['qx']),label=str(list_Q_L_round[i])+' L/h')
        plt.legend()

    plt.xlabel('N° riser')
    plt.ylabel('qx (L/h)')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()

    plt.show()

# Répartition des débits par canal (en valeur relative au débit moyen attendu)

def plot_flow_rates_harp(par,list_Q_L,list_PL,list_tabl,list_mn):

    list_Q_L_round = [round(num, 0) for num in list_Q_L]
    risers = np.linspace(0,par["N"]-1,par["N"])

    for i in range(0,len(list_tabl)):
        plt.plot(risers,np.array(list_tabl[i]['qx'])/list_mn[i],label=str(list_Q_L_round[i])+' L/h')

    plt.xlabel('N° riser')
    plt.ylabel('qx/q_mean')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()

    plt.show()