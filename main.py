import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.core.display import HTML
from IPython.display import display

import model2 as mod
import model_fsolve as modf

par = {}

par["eps"] = 0.001

par["ref"] = 1

par["rho"] = 997 # kg/m3
par["nu"] = 0.896*1e-6 # at 25°C, in m2/s https://www4.ac-nancy-metz.fr/physique/ancien_site/Tp-phys/Term/TP-fluid/visco-eau.htm
par["eta"] = par["rho"]*par["nu"]

# Heat exchanger inputs --------------------------------------------------

par["N"] = 100

par["Lx"] = 1.39985 # m, not used in the row calculation
par["Ly"] = 0.005 # m
par["Dx"] = 0.005 # m, not used in the row calculation
par["Din"] = 0.020 # m
par["Dout"] = 0.020 # m

par["rough"] = 0.0015 # PVC/plastic pipe absolute roughness is 0.0015

# ------------------------------------------------------------------------

par["Ax"] = math.pi*(par["Dx"]/2)**2
par["Ain"] = math.pi*(par["Din"]/2)**2
par["Aout"] = math.pi*(par["Dout"]/2)**2

par["a_x"] = 84.7/800
# par["a"] = 0.
par["b_x"] = 5.36/800

# dP (Pa) = (rho/2) (a_x u**2 + b_x u) its the pressure loss function of a heat exchanger

main = "liste"
sch = "exchanger"

if main == "solo":

    par["QF"] = 100/3600000 # m3/s (0.000278 m3/s = 1000 L/h)
    # Speed and Reynolds at inlet manifold
    par["U"] = par["QF"]/par["Ain"]
    par["Reman"] = par["U"]*(par["rho"]*par["Din"])/par["eta"]
    res = modf.PL_fsolve(par,sch,True)
    

elif main == "liste":
    
    # list_k = np.linspace(1e-6,5*1e-6,10)
    # list_k = np.linspace(1e-8,1e-6,20)
    list_Q = np.linspace(10/3600000,200/3600000,10)

    list_Q_L = []
    list_PL = []

    for Q in list_Q:
        print(Q)
        par["QF"] = Q
        # Speed and Reynolds at inlet manifold
        par["U"] = par["QF"]/par["Ain"]
        par["Reman"] = par["U"]*(par["rho"]*par["Din"])/par["eta"]

        res = modf.PL_fsolve(par,sch,False)
        list_Q_L.append(Q*3600000)
        list_PL.append(res)

    print(list_Q_L)
    print(list_PL)

    # df_res = pd.DataFrame([np.array(list_Q_L),np.array(list_PL)],columns = ['Q_L','PL (Pa)'])
    # display(HTML(df_res.to_html()))  

    plt.plot(np.array(list_Q_L),np.array(list_PL))

    plt.xlabel('Q (L/h)')
    plt.ylabel('PL (Pa)')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()

    plt.show()
else:
    pass