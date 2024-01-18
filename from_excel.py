import sys
import pandas as pd
import numpy as np
sys.path.append("../RD-systems-and-test-benches/utils")
import data_processing as dp
import hx_hydraulic as hxhy
from CoolProp.CoolProp import PropsSI
import model_fsolve as modf
import fluids as fds

def initialize(path, file_name):
    file_path = path+file_name

    hx = hxhy.hx_harp(dp.create_dict_from_excel(path,file_name,"heat_exchanger"))
    par = hx.make_dict()
    condi = pd.read_excel(file_path,sheet_name="conditions",header=0)
    condi0 = dp.create_dict_from_df_row(condi,0)
    condi0["p"] *= 1E5
    condi0["T"] += 273.15
    condi0["rho"] = PropsSI('D', 'P', condi0["p"], 'T', condi0["T"], f'INCOMP::{condi0["fluid"]}[{condi0["glycol_rate"]}]') # kg/m3
    condi0["eta"] = PropsSI('V', 'P', condi0["p"], 'T', condi0["T"], f'INCOMP::{condi0["fluid"]}[{condi0["glycol_rate"]}]')
    condi0["nu"] = condi0["eta"]/condi0["rho"]

    hx.compute_metrics()

    return(hx, par, condi0)


def read_excel(file_name):
    # Open the Excel file
    xls = pd.ExcelFile(file_name)

    # Read the 'Vdot_and_PL' sheet into a dataframe
    df_Vdot_PL = pd.read_excel(xls, 'PL')

    # Initialize lists for the table and AG table dataframes
    list_tabl = []
    list_agdf = []

    # Iterate over the remaining sheets in the Excel file
    for sheet_name in xls.sheet_names:
        if 'FD_by_channel_' in sheet_name:
            # Read the data from the 'Table' sheets into dataframes and append them to list_tabl
            df_table = pd.read_excel(xls, sheet_name)
            list_tabl.append(df_table)
        elif 'FD_by_panel_' in sheet_name:
            # Read the data from the 'AGTable' sheets into dataframes and append them to list_agdf
            df_agtable = pd.read_excel(xls, sheet_name)
            list_agdf.append(df_agtable)

    # Return the DataFrame and the lists of dataframes
    return [df_Vdot_PL, list_tabl, list_agdf]


def create_excel(list_Vdot, list_PL, list_tabl, list_agdf, file_name):
    # Create a new Excel writer object
    writer = pd.ExcelWriter(file_name, engine='openpyxl')

    # Create a dataframe for the Vdot and PL values and write it to the Excel file
    df_Vdot_PL = pd.DataFrame(list(zip(list_Vdot, list_PL)), columns=['Vdot', 'PL'])
    df_Vdot_PL.to_excel(writer, sheet_name='PL', index=False)

    # Iterate over the list of dataframes and write each one to a separate sheet in the Excel file
    for i, df in enumerate(list_tabl):
        df.to_excel(writer, sheet_name=f'FD_by_channel_{list_Vdot[i]}', index=False)

    # Iterate over the list_agdf and write each one to a separate sheet in the Excel file
    for i, df in enumerate(list_agdf):
        df.to_excel(writer, sheet_name=f'FD_by_panel_{list_Vdot[i]}', index=False)

    # Save the Excel file
    writer.close()

def testing_series(file_name, list_QF, list_QF_out, list_alpha, par,cond):
    rho = cond["rho"]
    eta = cond["eta"]
    ep = par["roughness"]
    Din = par["D_man"]
    Dout = par["D_man"]
    Dx = par["D_riser"]*np.sqrt(par['N'])
    Ax = par["A_riser"]*par['N']
    L_man = par["L_man"]
    L_riser = par["L_riser"]
    ref = par["ref"]

    df_QF = pd.DataFrame({'QF' : list_QF})
    df_QF_out = pd.DataFrame({'QF_out' : list_QF_out})
    df_alpha = pd.DataFrame({'alpha' : list_alpha})
    df_cond_testings = df_QF.merge(df_QF_out, how = 'cross').merge(df_alpha, how='cross')
    for i in range(len(df_cond_testings)):
        tabl, PL, df_PL, testings = modf.PL_fsolve(par,cond, series=list(df_cond_testings.loc[i]))
        N = len(tabl)
        df_cond_testings.loc[i,"Pin distrib"] = tabl.iloc[0]['Pin']
        df_cond_testings.loc[i,"Pout distrib"] = tabl.iloc[N-1]['Pin']
        df_cond_testings.loc[i,"Pin collect"] = tabl.iloc[ref]['Pout']
        df_cond_testings.loc[i,"Pout collect"] = tabl.iloc[N-1-ref]['Pout']

    # list_testings = []
    # for i in list(df_cond_testings.index):
    #     tabl, PL, df_PL, testings = modf.PL_fsolve(par,cond, series=list(df_cond_testings.loc[i]))
    #     list_testings.append(testings)

    # df_testings = pd.DataFrame(list_testings, columns=["Pin 1", "uin 1", "Pin 2", "uin 2", "Pout 1", "uout 1", "Pout 2", "uout 2"])
    # df_testings['ux'] = df_cond_testings['alpha']*df_cond_testings['QF']/Ax   

    # df_testings["Rein"] = fds.core.Reynolds((df_testings["uin 1"] + df_testings["uin 2"])/2,Din,rho,mu=eta)
    # df_testings["Reout"] = fds.core.Reynolds((df_testings["uout 1"] + df_testings["uout 2"])/2,Dout,rho,mu=eta)
    # df_testings["Rex"] = fds.core.Reynolds(df_testings["ux"],Dx,rho,mu=eta)
    # df_testings["fin"] = [fds.friction.friction_factor(Re = df_testings.loc[i]["Rein"],eD = ep/Din) for i in range(len(df_testings))]
    # df_testings["fout"] = [fds.friction.friction_factor(Re = df_testings.loc[i]["Reout"],eD = ep/Dout) for i in range(len(df_testings))]
    # df_testings["fx"] = [fds.friction.friction_factor(Re = df_testings.loc[i]["Rex"],eD = ep/Dx) for i in range(len(df_testings))]

    # df_testings["DPin"] = df_testings["Pin 1"] - df_testings["Pin 2"] + (rho/2) * (df_testings["uin 1"]**2 - df_testings["uin 2"]**2 - df_testings["fin"]*L_man*df_testings["uin 2"]**2/Din)
    # df_testings["DPout"] = df_testings["Pout 1"] - df_testings["Pout 2"] + (rho/2)*(df_testings["uout 1"]**2 - df_testings["uout 2"]**2 - df_testings["fout"]*L_man*df_testings["uout 2"]**2/Dout)
    # df_testings["DPx"] = df_testings["Pin 1"]- df_testings["Pout 2"] + (rho/2)*(df_testings["uin 1"]**2 - df_testings["uout 2"]**2 - df_testings["fx"]*L_riser*df_testings["ux"]**2/Dx)

    # df_testings = df_cond_testings.join(df_testings[["DPin","DPout","DPx"]])
    # df_testings.to_excel(file_name[:-5]+"_test.xlsx", index=False)
    return(df_cond_testings)

def testing_series_Qmax(Qmax, list_QF, list_alpha, par,cond):
    rho = cond["rho"]
    eta = cond["eta"]
    ep = par["roughness"]
    Din = par["D_man"]
    Dout = par["D_man"]
    Dx = par["D_riser"]*np.sqrt(par['N'])
    Ax = par["A_riser"]*par['N']
    L_man = par["L_man"]
    L_riser = par["L_riser"]
    ref = par["ref"]

    df_QF = pd.DataFrame({'Qin distrib' : list_QF})
    df_QF['Qin collect'] = Qmax - df_QF['Qin distrib']
    df_alpha = pd.DataFrame({'alpha' : list_alpha})
    df_cond_testings = df_QF.merge(df_alpha, how='cross')
    for i in range(len(df_cond_testings)):
        tabl, PL, df_PL, testings = modf.PL_fsolve(par,cond, series=list(df_cond_testings.loc[i]))
        N = len(tabl)
        df_cond_testings.loc[i,"Pin distrib"] = tabl.iloc[0]['Pin']
        df_cond_testings.loc[i,"Pout distrib"] = tabl.iloc[N-1]['Pin']
        df_cond_testings.loc[i,"Pin collect"] = tabl.iloc[ref]['Pout']
        df_cond_testings.loc[i,"Pout collect"] = tabl.iloc[N-1-ref]['Pout']

    return(df_cond_testings)