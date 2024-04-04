import sys
sys.path.append("../RD-systems-and-test-benches/utils")
import pandas as pd
import data_processing as dp
import hx_hydraulic as hxhy
from CoolProp.CoolProp import PropsSI
import model_fsolve as modf


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


def testing_series_Qmax(Qmax, list_Qin, list_alpha, par, cond, to_excel=False, file_name=None):
    ref = par["ref"]

    df_QF = pd.DataFrame({'Qin_d' : list_Qin})
    df_QF['Qin_c'] = Qmax - df_QF['Qin_d']
    df_alpha = pd.DataFrame({'alpha' : list_alpha})
    df_cond_testings = df_QF.merge(df_alpha, how='cross')
    for i in range(len(df_cond_testings)):
        tabl, PL, df_PL, residuals = modf.PL_fsolve(par,cond, series=list(df_cond_testings.loc[i]))
        N = len(tabl)
        df_cond_testings.loc[i,"Pin_d"] = tabl.iloc[0]['Pin']
        df_cond_testings.loc[i,"Pout_d"] = tabl.iloc[N-1]['Pin']
        df_cond_testings.loc[i,"Pin_c"] = tabl.iloc[ref]['Pout']
        df_cond_testings.loc[i,"Pout_c"] = tabl.iloc[N-1-ref]['Pout']

    if to_excel:
        df_cond_testings.to_excel(file_name)

    return(df_cond_testings)