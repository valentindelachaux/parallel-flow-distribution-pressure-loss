import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def process_xy_file(folder_path,file_name, x_name, y_name):
    df = pd.read_csv(f'{folder_path}\\fluent_export\\{file_name}')
    df[['Column1', 'Column2']] = df[df.columns[0]].str.split('\t', expand=True)
    df = df.iloc[2:-1].drop(columns=[df.columns[0]]).rename(columns={'Column1': x_name, 'Column2': y_name}).astype(float)
    df = df.reset_index(drop=True)
    df = df.sort_values(by=x_name, ascending=True)
    df[x_name] = df[x_name] - df[x_name].min()
    df.to_csv(folder_path+'\\processed\\'+file_name, index=False)

width_all_risers = 0.2153
w_riser = 0.00281
inter_EP = 0.0047
inter_riser = 0.00035
N_riser_per_EP = 16
N_EP = 4

# Initialize the list
values_list = [0]

# Use a loop to generate the list values
for i in range(1,32*4):  # Generating 64 values as the example ends at 64th value

    if i%2 == 1:
        value = values_list[i-1] + w_riser
    else:
        value = values_list[i-1] + inter_riser

    if i % 32 == 0:
        value += (inter_EP - inter_riser)
        
    # Append the value to the list
    values_list.append(value)

def check_value(value):
    for i in range(len(values_list)-1):
        if i % 2 == 0 and values_list[i] <= value <= values_list[i+1]:
            return 1
    return 0

def process_pressure_risers(df):
    dfp = pd.DataFrame(np.arange(df['Position'].min(),df['Position'].max(),0.00001),columns=['Position'])

    concatenated_df = pd.concat([df, dfp])
    df = concatenated_df.sort_values(by='Position')
    df.reset_index(drop=True,inplace=True)

    df["check"] = df['Position'].apply(lambda x : check_value(x))

    df['Group'] = (df['check'] != df['check'].shift()).cumsum() * df['check']
    df['Group'] = df['Group'].apply(lambda x : x//2 if x!=0 else -1)
    pressure_risers = df.groupby('Group')['Pressure'].mean()

    pressure_risers = pressure_risers.drop(pressure_risers.index[0])
    pressure_risers = pd.DataFrame(pressure_risers)

    return df,pressure_risers