import math
from re import A
import pandas as pd
import numpy as np
from IPython.core.display import HTML
from IPython.display import display
import fluids.fittings as ffit
import scipy.optimize as sc

rho = 997
g = 9.81

def find_PL(par,debug):

    data_columns = ['Qin','qx','qx_new','Qout','uin','ux','uout','Rein','Rex','Reout','fyin','fx','fyout','Kxin','Kxout','Kyin','Kyout','Pin','Pout','DPx','DPin','DPout','alpha','ERR_terms']

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

    rho = par["rho"]

    # par["Reman"] is initialized in "main.py"

    for i in range(1,par["N"]+1):
        if par["Reman"]<2100:
            df.loc[i,"fin"] = 64/par["Reman"] # laminar flow with Darcy-Weisbach friction factor
            df.loc[i,"fout"] = 64/par["Reman"] # laminar flow with Darcy-Weisbach friction factor
        else:
            df.loc[i,"fin"] = 0.3164*par["Reman"]**(-0.25) 
            df.loc[i,"fout"] = 0.3164*par["Reman"]**(-0.25) # Blasius equation for turbulent flow

    # Step 1 : initial guess of cell flow rates and a reference inlet pressure drop

    for i in range(1,par["N"]+1):
        # df.loc[ref,"qx"] = QF - par["N"]*0.000001
        # if i > ref:
        #     df.loc[i,"qx"] = 0.000001
        df.loc[i,"qx"] = QF/par["N"]
        df.loc[i,'ux'] = df.loc[i,"qx"]/Ax
        df.loc[i,"Rex"] = df.loc[i,"ux"]*(rho*Dx)/eta # eta = par["eta"]

        # Calculation of fx(i)

        if df.loc[i,"Rex"]<2100:
            df.loc[i,"fx"] = 64/df.loc[i,"Rex"] # laminar flow with Darcy-Weisbach friction factor
        else:
            df.loc[i,"fx"] = 0.3164*df.loc[i,"Rex"]**(-0.25) # Blasius equation for turbulent flow

    # Calculation of flow rates along y

    for i_ in range(1,par["N"]+1):
        df.loc[i_,"Qin"] = df["qx"][0:i_].sum()
        df.loc[i_,"Qout"] = df["qx"][i_-1:N].sum() 

    df.loc[ref,'Kxin'] = ffit.K_branch_diverging_Crane(D_run=Din, D_branch=Dx, Q_run=df.loc[ref,'Qin'], Q_branch=df.loc[ref,'qx'], angle=90.)
    # Kxin(kxin,df.loc[ref,'ux'],df.loc[ref,'uin']) # wb et wc
    df.loc[ref,'Kxout'] = ffit.K_branch_converging_Crane(D_run=Dout, D_branch=Dx, Q_run=df.loc[ref,'Qout'], Q_branch=df.loc[ref,'qx'], angle=90.)
    # Kxout(kxout,df.loc[ref,'qx'],df.loc[ref,'Qout'],Ax,Aout) # Qb, Qc, Fb, Fc

    df.loc[ref,"DPx"] = (rho/2)*(df.loc[ref,"fx"]*(Lx/Dx)*df.loc[ref,'ux']**2+df.loc[ref,'Kxin']*df.loc[ref,'uin']**2+df.loc[ref,'Kxout']*df.loc[ref,'uout']**2)

    ERR = 1
    time = 0

    if debug==True:
        print("Initial conditions")
        display(HTML(df.to_html()))  

    while ERR>eps:
        time+=1

        # Step 2

        for i_ in range(1,par["N"]+1):

            # for slicing, it is counted from 0 to N-1, for [a:b], b is excluded
            df.loc[i_,"Qin"] = df["qx"][0:i_].sum()
            df.loc[i_,"Qout"] = df["qx"][i_-1:N].sum() # for parallel flow

        for i in range(1,par["N"]+1):
            # Linear velocities in manifolds (uin and uout) and risers (ux)
            df.loc[i,"uin"] = df.loc[i,"Qin"]/Ain
            df.loc[i,"ux"] = df.loc[i,"qx"]/Ax
            df.loc[i,"uout"] = df.loc[i,"Qout"]/Aout
            
            # Reynolds numbers
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

            # if df.loc[i,"Reout"]<2100:
            #     df.loc[i,"fout"] = 64/df.loc[i,"Reout"] # laminar flow with Darcy-Weisbach friction factor
            # else:
            #     df.loc[i,"fout"] = 0.3164*df.loc[i,"Reout"]**(-0.25) # Blasius equation for turbulent flow

        # Step 3 : Calculate outlet manifold pressure

        Pexit = 0
        df.loc[ref,"Pout"] = Pexit

        for i in range(1,par["N"]+1):
            df.loc[i,'Kxin'] = ffit.K_branch_diverging_Crane(D_run=Din, D_branch=Dx, Q_run=df.loc[i,'Qin'], Q_branch=df.loc[i,'qx'], angle=90.)
            # df.loc[i,'Kxin'] = Kxin(kxin,df.loc[i,'ux'],df.loc[i,'uin']) # wb and wc
            if i == 1:
                df.loc[i,'Kyin'] = 0.
            else:
                df.loc[i,'Kyin'] = ffit.K_run_diverging_Crane(D_run=Din, D_branch=Dx, Q_run=df.loc[i-1,'Qout'], Q_branch=df.loc[i,'qx'], angle=90.)
                # df.loc[i,'Kyin'] = Kyin(kyin,df.loc[i-1,'uin'],df.loc[i,'uin']) # ws and wc
            
            df.loc[i,'Kxout'] = ffit.K_branch_converging_Crane(D_run=Dout, D_branch=Dx, Q_run=df.loc[i,'Qout'], Q_branch=df.loc[i,'qx'], angle=90.)
            # df.loc[i, 'Kxout'] = Kxout(kxout,df.loc[i,'qx'],df.loc[i,'Qout'],Ax,Aout) # Qb, Qc, Fb, Fc
            
            df.loc[i, 'Kyout'] = ffit.K_run_converging_Crane(D_run=Dout, D_branch=Dx, Q_run=df.loc[i,'Qout'], Q_branch=df.loc[i,'qx'], angle=90.)
            # df.loc[i,'Kyout'] = Kyout(kyout,df.loc[i,'qx'],df.loc[i,'Qout']) # Qb and Qc

        for i in range(2,par["N"]+1): 
            df.loc[i,"DPout"] = (rho/2)*(df.loc[i-1,"uout"]**2-df.loc[i,"uout"]**2 + (df.loc[i,"fout"]*Ly*df.loc[i,"uout"]**2)/Dout + df.loc[i,'Kyout']*df.loc[i-1,'uout']**2)
            df.loc[i,"Pout"] = df.loc[i-1,"Pout"] + df.loc[i,"DPout"] # for parallel flow

        # Step 4 : Calculate inlet manifold pressure

        df.loc[ref,"Pin"] = df.loc[ref,"DPx"]+df.loc[ref,"Pout"]

        for i in range(2,par["N"]+1):
            df.loc[i,"DPin"] = (rho/2)*(df.loc[i-1,"uin"]**2-df.loc[i,"uin"]**2 + (df.loc[i,"fin"]*Ly*df.loc[i,"uin"]**2)/Din + df.loc[i,'Kyin']*df.loc[i,'uin']**2)
            df.loc[i,"Pin"] = df.loc[i-1,"Pin"] + df.loc[i,"DPin"] # for parallel or reverse flow

        # Step 5 : Calculate flow distribution factors

        for i in range(1,par["N"]+1):
            df.loc[i,"alpha"] = (df.loc[i,"Pin"]-df.loc[i,"Pout"])/df.loc[ref,"DPx"]

        # Step 6 : Calculate a new value for the pressure drop over the first channel

        sum_DPxi = 0
        sum_alphai = 0

        for i in range(1,par["N"]+1):
            sum_DPxi += (rho/2)*(df.loc[i,"fx"]*(Lx/Dx)*df.loc[i,'ux']**2+df.loc[i,'Kxin']*df.loc[i,'uin']**2+df.loc[i,'Kxout']*df.loc[i,'uout']**2)
            sum_alphai += df.loc[i,"alpha"]
        
        df.loc[ref,"DPx"] = sum_DPxi/sum_alphai

        # Step 7

        df.loc[ref,"Pin"] = df.loc[ref,"DPx"] + df.loc[ref,"Pout"]

        for i in range(2,par["N"]+1):
            df.loc[i,"Pin"] = df.loc[i-1,"Pin"] + df.loc[i,"DPin"] # for parallel or reverse flow

        # Step 8 : Adjust flow distribution factors from new pressure drop over the first channel

        for i in range(1,par["N"]+1): # i = 1, ..., N
            df.loc[i,"alpha"] = (df.loc[i,"Pin"]-df.loc[i,"Pout"])/df.loc[ref,"DPx"]

        flag = 0
        # Step 9
        for i in range(1,par["N"]+1):

            if (Dx/(df.loc[i,"fx"]*Lx))*((2/rho)*(df.loc[i,"Pin"]-df.loc[i,"Pout"])-df.loc[i,'Kxin']*df.loc[i,'uin']**2-df.loc[i,'Kxout']*df.loc[i,'uout']**2)<=0:
                flag = 1
            else:
                pass
        
        if flag == 1:
            for i in range(1,par["N"]+1):
                df.loc[i,"qx_new"] = df.loc[i,"qx"]
        else:
            for i in range(1,par["N"]+1):
                df.loc[i,"qx_new"] = Ax*math.sqrt((Dx/(df.loc[i,"fx"]*Lx))*((2/rho)*(df.loc[i,"Pin"]-df.loc[i,"Pout"])-df.loc[i,'Kxin']*df.loc[i,'uin']**2-df.loc[i,'Kxout']*df.loc[i,'uout']**2))

        
        # Step 10 : Check convergence

        for i in range(1,par["N"]+1):
            df.loc[i,"ERR_terms"] = ((df.loc[i,"qx_new"]-df.loc[i,"qx"])/df.loc[i,"qx"])**2
            df.loc[i,"qx"] = df.loc[i,"qx_new"]
        
        ERR = df["ERR_terms"].sum()

        if debug == True:
            print("End of "+str(time)+"th step")
            display(HTML(df.to_html()))    

    # DPx
    for i in range(2,par["N"]+1):
        df.loc[i,"DPx"] = df.loc[i,"Pin"] - df.loc[i,"Pout"]

    if debug == True:
        print("End")
        display(HTML(df.to_html()))  

    return df["Pin"][N]

def row_find_PL(par,debug): # par doit avoir les coeffs a et b devant x2 et x du panneau

    data_columns = ['Qin','qx','qx_new','Qout','uin','ux','uout','Rein','Rex','Reout','fyin','ax','bx','fyout','Kxin','Kxout','Kyin','Kyout','Pin','Pout','DPx','DPin','DPout','alpha','ERR_terms']

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

    rho = par["rho"]

    # par["Reman"] is initialized in "main.py"

    for i in range(1,par["N"]+1):
        if par["Reman"]<2100:
            df.loc[i,"fin"] = 64/par["Reman"] # laminar flow with Darcy-Weisbach friction factor
            df.loc[i,"fout"] = 64/par["Reman"] # laminar flow with Darcy-Weisbach friction factor
        else:
            df.loc[i,"fin"] = 0.3164*par["Reman"]**(-0.25) 
            df.loc[i,"fout"] = 0.3164*par["Reman"]**(-0.25) # Blasius equation for turbulent flow
        df.loc[i,'ax']=par["a"]
        df.loc[i,'bx']=par["b"]

    # Step 1 : initial guess of cell flow rates and a reference inlet pressure drop

    for i in range(1,par["N"]+1):
        df.loc[ref,"qx"] = QF - (par["N"]-1)*0.000001
        if i > ref:
            df.loc[i,"qx"] = 0.000001
        df.loc[i,'ux'] = df.loc[i,"qx"]/Ax
        df.loc[i,"Rex"] = df.loc[i,"ux"]*(rho*Dx)/eta # eta = par["eta"]

        # Calculation of fx(i)

        if df.loc[i,"Rex"]<2100:
            df.loc[i,"fx"] = 64/df.loc[i,"Rex"] # laminar flow with Darcy-Weisbach friction factor
        else:
            df.loc[i,"fx"] = 0.3164*df.loc[i,"Rex"]**(-0.25) # Blasius equation for turbulent flow

    # Calculation of flow rates along y

    for i_ in range(1,par["N"]+1):
        df.loc[i_,"Qin"] = df["qx"][0:i_].sum()
        df.loc[i_,"Qout"] = df["qx"][i_-1:N].sum() 

    df.loc[ref,'Kxin'] = ffit.K_branch_diverging_Crane(D_run=Din, D_branch=Dx, Q_run=df.loc[ref,'Qin'], Q_branch=df.loc[ref,'qx'], angle=90.)
    # Kxin(kxin,df.loc[ref,'ux'],df.loc[ref,'uin']) # wb et wc
    df.loc[ref,'Kxout'] = ffit.K_branch_converging_Crane(D_run=Dout, D_branch=Dx, Q_run=df.loc[ref,'Qout'], Q_branch=df.loc[ref,'qx'], angle=90.)
    # Kxout(kxout,df.loc[ref,'qx'],df.loc[ref,'Qout'],Ax,Aout) # Qb, Qc, Fb, Fc

    df.loc[ref,"DPx"] = np.polyval([df.loc[ref,"ax"],df.loc[ref,"bx"],0],df.loc[ref,'ux'])+(rho/2)*(df.loc[ref,'Kxin']*df.loc[ref,'uin']**2+df.loc[ref,'Kxout']*df.loc[ref,'uout']**2)

    ERR = 1
    time = 0

    if debug==True:
        print("Initial conditions")
        display(HTML(df.to_html()))  

    while ERR>eps:
        time+=1

        # Step 2

        for i_ in range(1,par["N"]+1):

            # for slicing, it is counted from 0 to N-1, for [a:b], b is excluded
            df.loc[i_,"Qin"] = df["qx"][0:i_].sum()
            df.loc[i_,"Qout"] = df["qx"][i_-1:N].sum() # for parallel flow

        for i in range(1,par["N"]+1):
            # Linear velocities in manifolds (uin and uout) and risers (ux)
            df.loc[i,"uin"] = df.loc[i,"Qin"]/Ain
            df.loc[i,"ux"] = df.loc[i,"qx"]/Ax
            df.loc[i,"uout"] = df.loc[i,"Qout"]/Aout
            
            # Reynolds numbers
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

            # if df.loc[i,"Reout"]<2100:
            #     df.loc[i,"fout"] = 64/df.loc[i,"Reout"] # laminar flow with Darcy-Weisbach friction factor
            # else:
            #     df.loc[i,"fout"] = 0.3164*df.loc[i,"Reout"]**(-0.25) # Blasius equation for turbulent flow

        # Step 3 : Calculate outlet manifold pressure

        Pexit = 0
        df.loc[1,"Pout"] = Pexit

        for i in range(1,par["N"]+1):
            df.loc[i,'Kxin'] = ffit.K_branch_diverging_Crane(D_run=Din, D_branch=Dx, Q_run=df.loc[i,'Qin'], Q_branch=df.loc[i,'qx'], angle=90.)
            # df.loc[i,'Kxin'] = Kxin(kxin,df.loc[i,'ux'],df.loc[i,'uin']) # wb and wc
            if i == 1:
                df.loc[i,'Kyin'] = 0.
            else:
                df.loc[i,'Kyin'] = ffit.K_run_diverging_Crane(D_run=Din, D_branch=Dx, Q_run=df.loc[i-1,'Qout'], Q_branch=df.loc[i,'qx'], angle=90.)
                # df.loc[i,'Kyin'] = Kyin(kyin,df.loc[i-1,'uin'],df.loc[i,'uin']) # ws and wc
            
            df.loc[i,'Kxout'] = ffit.K_branch_converging_Crane(D_run=Dout, D_branch=Dx, Q_run=df.loc[i,'Qout'], Q_branch=df.loc[i,'qx'], angle=90.)
            # df.loc[i, 'Kxout'] = Kxout(kxout,df.loc[i,'qx'],df.loc[i,'Qout'],Ax,Aout) # Qb, Qc, Fb, Fc
            
            df.loc[i, 'Kyout'] = ffit.K_run_converging_Crane(D_run=Dout, D_branch=Dx, Q_run=df.loc[i,'Qout'], Q_branch=df.loc[i,'qx'], angle=90.)
            # df.loc[i,'Kyout'] = Kyout(kyout,df.loc[i,'qx'],df.loc[i,'Qout']) # Qb and Qc

        for i in range(2,par["N"]+1): 
            df.loc[i,"DPout"] = (rho/2)*(df.loc[i-1,"uout"]**2-df.loc[i,"uout"]**2 + (df.loc[i,"fout"]*Ly*df.loc[i,"uout"]**2)/Dout + df.loc[i,'Kyout']*df.loc[i-1,'uout'])
            df.loc[i,"Pout"] = df.loc[i-1,"Pout"] + df.loc[i,"DPout"] # for parallel flow

        # Step 4 : Calculate inlet manifold pressure

        df.loc[ref,"Pin"] = df.loc[ref,"DPx"]+df.loc[ref,"Pout"]

        for i in range(2,par["N"]+1):
            df.loc[i,"DPin"] = (rho/2)*(df.loc[i-1,"uin"]**2-df.loc[i,"uin"]**2 + (df.loc[i,"fin"]*Ly*df.loc[i,"uin"]**2)/Din + df.loc[i,'Kyin']*df.loc[i,'uin']**2)
            df.loc[i,"Pin"] = df.loc[i-1,"Pin"] + df.loc[i,"DPin"] # for parallel or reverse flow

        # Step 5 : Calculate flow distribution factors

        for i in range(1,par["N"]+1):
            df.loc[i,"alpha"] = (df.loc[i,"Pin"]-df.loc[i,"Pout"])/df.loc[ref,"DPx"]

        # Step 6 : Calculate a new value for the pressure drop over the first channel

        sum_DPxi = 0
        sum_alphai = 0

        for i in range(1,par["N"]+1):
            sum_DPxi += np.polyval([df.loc[ref,"ax"],df.loc[ref,"bx"],0],df.loc[ref,'ux'])+(rho/2)*(df.loc[i,'Kxin']*df.loc[i,'uin']**2+df.loc[i,'Kxout']*df.loc[i,'uout']**2)
            sum_alphai += df.loc[i,"alpha"]
        
        df.loc[ref,"DPx"] = sum_DPxi/sum_alphai

        # Step 7

        df.loc[ref,"Pin"] = df.loc[ref,"DPx"] + df.loc[ref,"Pout"]

        for i in range(2,par["N"]+1):
            df.loc[i,"Pin"] = df.loc[i-1,"Pin"] + df.loc[i,"DPin"] # for parallel or reverse flow

        # Step 8 : Adjust flow distribution factors from new pressure drop over the first channel

        for i in range(1,par["N"]+1): # i = 1, ..., N
            df.loc[i,"alpha"] = (df.loc[i,"Pin"]-df.loc[i,"Pout"])/df.loc[ref,"DPx"]

        flag = 0
        # Step 9
        for i in range(1,par["N"]+1):

            res = sc.fsolve(lambda v : df.loc[i,"ax"]*v**2 + df.loc[i,"bx"]*v + (rho/2)*(df.loc[i,'Kxin']*df.loc[i,'uin']**2+df.loc[i,'Kxout']*df.loc[i,'uout']**2) - (df.loc[i,"Pin"]-df.loc[i,"Pout"]),df.loc[i,"qx"])
            l = len(res)
            
            if l == 0:
                df.loc[i,"qx_new"] = df.loc[i,'qx']
            elif l == 1:
                if res[0] > 0:
                    df.loc[i,"qx_new"] = res[0]
                else:
                   df.loc[i,"qx_new"] = df.loc[i,'qx']
            else: 
                if res[0] > 0 and res[1]>0:
                    if abs(df.loc[i,'qx']-res[0]) < abs(df.loc[i,'qx']-res[0]):
                        df.loc[i,"qx_new"] = res[0]
                    else:
                        df.loc[i,"qx_new"] = res[1]
                elif res[0] > 0 and res[1] < 0:
                    df.loc[i,"qx_new"] = res[0]
                elif res[0] < 0 and res[1] > 0:
                    df.loc[i,"qx_new"] = res[1]
                else:
                    df.loc[i,"qx_new"] = df.loc[i,'qx']
       
        # Step 10 : Check convergence

        for i in range(1,par["N"]+1):
            df.loc[i,"ERR_terms"] = ((df.loc[i,"qx_new"]-df.loc[i,"qx"])/df.loc[i,"qx"])**2
            df.loc[i,"qx"] = df.loc[i,"qx_new"]
        
        ERR = df["ERR_terms"].sum()

        if debug == True:
            print("End of "+str(time)+"th step")
            display(HTML(df.to_html()))    

    # DPx
    for i in range(2,par["N"]+1):
        df.loc[i,"DPx"] = df.loc[i,"Pin"] - df.loc[i,"Pout"]

    if debug == True:
        print("End")
        display(HTML(df.to_html()))  

    return df["Pin"][N]
