import math as mt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fluids.fittings as ffit

import model2 as mod

Drun = 0.042 # m
Dbranch = 0.008 # m
Arun = 0.25*mt.pi*Drun**2
Abranch = 0.25*mt.pi*Dbranch**2

beta = Dbranch/Drun

rho = 997

Q_list = np.linspace(0.00001,0.0005,100)
hlb = []
hlr = []
hlb2 = []
hlr2 = []

for i in range(len(Q_list)):

    Q_straight = (9/10)*Q_list[i]
    Q_branch = (1/10)*Q_list[i]
    Q_tot = Q_straight + Q_branch

    v_tot = Q_tot/Arun
    v_straight = Q_straight/Arun
    v_branch = Q_branch/Abranch

    branch_flow_ratio = Q_branch/Q_tot

    K_branch = ffit.K_branch_diverging_Crane(D_run=Drun, D_branch=Dbranch, Q_run=Q_straight, Q_branch=Q_branch, angle=90.)
    K_run = ffit.K_run_diverging_Crane(D_run=Drun, D_branch=Dbranch, Q_run=Q_straight, Q_branch=Q_branch, angle=90.)

    head_loss_branch = 0.5*rho*v_tot**2*K_branch # Pa
    head_loss_run = 0.5*rho*v_tot**2*K_run # Pa

    hlb.append(head_loss_branch)
    hlr.append(head_loss_run)

    head_loss_branch_2 = 0.5*rho*v_tot**2*mod.Kxin(1.,v_branch,v_tot)
    head_loss_run_2 = 0.5*rho*v_tot**2*mod.Kyin(1.,v_straight,v_tot)

    hlb2.append(head_loss_branch_2)
    hlr2.append(head_loss_run_2)

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 1)

axis[0].plot(np.array(Q_list),np.array(hlb),label = "hlb")
axis[0].plot(np.array(Q_list),np.array(hlr),label = "hlr")

axis[0].legend()

plt.grid()

axis[1].plot(np.array(Q_list),np.array(hlb2),label="hlb2")
axis[1].plot(np.array(Q_list),np.array(hlr2),label="hlr2")

axis[1].legend()

# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()

plt.show()