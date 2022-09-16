import math
import pandas as pd
import numpy as np


def find_PL(par):

    data_columns = ['Qin','qx','qx_new','Qout','uin','ux','uout','Rein','Rex','Reout','fin','fx','fout','Pin','Pout','DPx','DPin','DPout','alpha','ERR_terms']

    uns = len(data_columns)*[1]
    liste = [uns for x in range(par["N"])]

    df = pd.DataFrame(liste, index = range(1,par["N"]+1),columns = data_columns)

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
    Ly = par["Ly"]
    eta = par["eta"] # dynamic viscosity
    k = par["k"] 
    rho = par["rho"]

    Kf = par["Kf"]

    for i in range(1,par["N"]+1):
        if par["Reman"]<2100:
            df.loc[i,"fin"] = 64/par["Reman"] # laminar flow with Darcy-Weisbach friction factor
            df.loc[i,"fout"] = 64/par["Reman"] # laminar flow with Darcy-Weisbach friction factor
        else:
            df.loc[i,"fin"] = 0.3164*par["Reman"]**(-0.25) 
            df.loc[i,"fout"] = 0.3164*par["Reman"]**(-0.25) # Blasius equation for turbulent flow

    # Step 1 : initial guess of cell flow rates and a reference inlet pressure drop

    for i in range(1,par["N"]+1):
        df.loc[i,"qx"] = QF/N
        df.loc[i,'ux'] = df.loc[i,"qx"]/Ax
        df.loc[i,"Rex"] = df.loc[i,"ux"]*(rho*Dx)/eta

        # Calculation of fx

        if df.loc[i,"Rex"]<2100:
            df.loc[i,"fx"] = 64/df.loc[i,"Rex"] # laminar flow with Darcy-Weisbach friction factor
        else:
            df.loc[i,"fx"] = 0.3164*df.loc[i,"Rex"]**(-0.25) # Blasius equation for turbulent flow


    # df.loc[ref,"DPx"] = (QF/N)*(eta/Ax)*(Lx/k)
    df.loc[ref,"DPx"] = df.loc[ref,"fx"]*(Lx/Dx)*rho*(1/2)*(U/N)**2

    ERR = 1
    time = 0

    # print(df)

    while ERR>eps:
        time+=1
        # print(time)

        # Step 2

        for i_ in range(1,par["N"]+1):

            # for slicing, it is counted from 0 to N-1, for [a:b], b is excluded
            df.loc[i_,"Qin"] = df["qx"][0:i_].sum()
            df.loc[i_,"Qout"] = df["qx"][i_-1:N].sum() # for parallel flow

        for i in range(1,par["N"]+1):
            df.loc[i,"uin"] = df.loc[i,"Qin"]/Ain
            df.loc[i,"ux"] = df.loc[i,"qx"]/Ax
            df.loc[i,"uout"] = df.loc[i,"Qout"]/Aout
                
            df.loc[i,"Rein"] = df.loc[i,"uin"]*(rho*Din)/eta
            df.loc[i,"Rex"] = df.loc[i,"ux"]*(rho*Dx)/eta
            df.loc[i,"Reout"] = df.loc[i,"uout"]*(rho*Dout)/eta

            # Calculation of fx(i) factors

            if df.loc[i,"Rex"]<2100:
                df.loc[i,"fx"] = 64/df.loc[i,"Rex"] # laminar flow with Darcy-Weisbach friction factor
            else:
                df.loc[i,"fx"] = 0.3164*df.loc[i,"Rex"]**(-0.25) # Blasius equation for turbulent flow
            
            # Calculation of fin(i) and fout(i) factors

            # if df.loc[i,"Rein"]<2100:
            #     df.loc[i,"fin"] = 64/df.loc[i,"Rein"] # laminar flow with Darcy-Weisbach friction factor
            # else:
            #     df.loc[i,"fin"] = 0.3164*df.loc[i,"Rein"]**(-0.25) # Blasius equation for turbulent flow

        # Step 3 : Calculate outlet manifold pressure

        Pexit = 0
        df.loc[1,"Pout"] = Pexit

        for i in range(2,par["N"]+1):
            df.loc[i,"DPout"] = rho*((1/2)*(df.loc[i-1,"uout"]**2-df.loc[i,"uout"]**2) + ((2*df.loc[i,"fout"]*Ly)/Dout + Kf/2)*df.loc[i,"uout"]**2)
            df.loc[i,"Pout"] = df.loc[i-1,"Pout"] + df.loc[i,"DPout"] # for parallel flow

        # Step 4 : Calculate inlet manifold pressure

        df.loc[ref,"Pin"] = df.loc[ref,"DPx"]+df.loc[ref,"Pout"]

        for i in range(2,par["N"]+1):
            df.loc[i,"DPin"] = rho*((1/2)*(df.loc[i-1,"uin"]**2-df.loc[i,"uin"]**2) + ((2*df.loc[i,"fin"]*Ly)/Din + Kf/2)*df.loc[i,"uin"]**2)
            df.loc[i,"Pin"] = df.loc[i-1,"Pin"] + df.loc[i,"DPin"] # for parallel or reverse flow
        # Step 5 : Calculate flow distribution factors

        for i in range(1,par["N"]+1):
            df.loc[i,"alpha"] = (df.loc[i,"Pin"]-df.loc[i,"Pout"])/df.loc[ref,"DPx"]

        # Step 6

        df.loc[ref,"DPx"] = (QF*eta*Lx)/(Ax*k*df["alpha"].sum())

        # df.loc[ref,"DPx"] = df.loc[i,"fx"]*(Lx/Dx)*rho*(1/2)*(U/df["alpha"].sum())**2

        # Step 7

        df.loc[ref,"Pin"] = df.loc[ref,"DPx"] + df.loc[ref,"Pout"]

        for i in range(2,par["N"]+1):
            df.loc[i,"Pin"] = df.loc[i-1,"Pin"] + df.loc[i,"DPin"] # for parallel or reverse flow

        # Step 8

        for i in range(1,par["N"]+1):
            df.loc[i,"alpha"] = (df.loc[i,"Pin"]-df.loc[i,"Pout"])/df.loc[ref,"DPx"]

        # Step 9
        for i in range(1,par["N"]+1):
            df.loc[i,"qx_new"] = ((k*Ax)/(eta*Lx))*(df.loc[i,"Pin"]-df.loc[i,"Pout"])

        # Step 10 : Check convergence

        for i in range(1,par["N"]+1):
            df.loc[i,"ERR_terms"] = ((df.loc[i,"qx_new"]-df.loc[i,"qx"])/df.loc[i,"qx"])**2
            df.loc[i,"qx"] = df.loc[i,"qx_new"]
        
        ERR = df["ERR_terms"].sum()
        # print(ERR)
        # print(df)     

    # DPx
    for i in range(2,par["N"]+1):
        df.loc[i,"DPx"] = df.loc[i,"Pin"] - df.loc[i,"Pout"]

    # print(df)

    # print(df["Pin"][N])

    return df["Pin"][N]

