import os 
import numpy as np
import pandas as pd

import openpyxl

import matplotlib.pyplot as plt


import jou_gen as jg

from io import StringIO

import plotly.graph_objects as go

def create_plot_x1_x2(df, X1_name, X2_name, Y_name):

    # Initialize a Plotly figure
    fig = go.Figure()

    # Get unique values of 'X2'
    unique_x2_values = df[X2_name].unique()

    # Loop over each unique value of 'X2' to create a line for each
    for value in unique_x2_values:
        # Filter the DataFrame for each value of 'X2'
        df_filtered = df[df[X2_name] == value]
        
        # Add a trace/line for the filtered DataFrame
        fig.add_trace(go.Scatter(x=df_filtered[X1_name], y=df_filtered[Y_name], mode='markers', name=f'{X2_name}={value}'))

    # Update layout if needed
    fig.update_layout(
        title=f'Plot of {Y_name} vs. {X1_name} for constant values of {X2_name}',
        xaxis_title=X1_name,
        yaxis_title=Y_name,
        legend_title=f'Values of {X2_name}'
    )

    # Show the figure
    return fig

def plot_Y_X(df, X_list, Y_list):

    fig_list = []

    for Y_name in Y_list:
        for i in range(len(X_list)):
            X1_name = X_list[i]
            X2_name = X_list[(i+1)%2]

            fig = create_plot_x1_x2(df, X1_name, X2_name, Y_name)

            poly_func_dict = interp_x1_x2(df, X1_name, X2_name, Y_name)
            unique_x2_values = df[X2_name].unique()

            # Loop over each unique value of X2_name to create a line for each
            for value in unique_x2_values:
                # Filter the DataFrame for each value of X2_name
                df_filtered = df[df[X2_name] == value]      
                fig.add_trace(go.Scatter(x=df_filtered[X1_name].unique(), y=poly_func_dict[f'{Y_name} at {X2_name} = {value}'](df_filtered[X1_name].unique()), mode='lines', name=f'Interpolated {Y_name} at {X2_name} = {value}'))

            fig_list.append(fig)

    return fig_list

def interp_x1_x2(df, X1_name, X2_name, Y_name):
        
    # Get unique values of 'X2'
    unique_x2_values = df[X2_name].unique()

    poly_coeffs_dict = {}
    poly_func_dict = {}

    # Loop over each unique value of 'X2' to create a line for each
    for value in unique_x2_values:
        # Filter the DataFrame for each value of 'X2'
        df_filtered = df[df[X2_name] == value]
    
        # Fit a degree 2 polynomial to the data
        poly_coeffs = np.polyfit(df_filtered[X1_name], df_filtered[Y_name], 3)
        
        # Create a polynomial function from the coefficients
        poly_func = np.poly1d(poly_coeffs)
        
        poly_func_dict[f'{Y_name} at {X2_name} = {value}'] = poly_func

    return poly_func_dict
