import numpy as np

def set_value(expr, key, dict, value):
    # Set the value of an expression in key of a dictionnary
    dict.loc[dict[key] == expr, 'value'] = value
    return

def get_value(expr, key, dict):
    # Get the value of an expression in key of a dictionnary
    return dict.loc[dict[key] == expr, 'value'].values[0]

def T_fluid_out(T_fluid_in, L, a_f, b_f):
    # Calculate the outlet temperature of the fluid
    T_fluid_out = (T_fluid_in + (b_f / a_f)) * np.exp(a_f * L) - (b_f / a_f)
    return T_fluid_out

def T_fluid_mean(T_fluid_in, L, a_f, b_f):
    # Calculate the mean temperature of the fluid
    T_fluid_mean = (T_fluid_in + (b_f / a_f))/(a_f*L) * np.exp(a_f * L) - (T_fluid_in + (b_f / a_f))/(a_f*L)- (b_f / a_f)
    return T_fluid_mean

def fill_initial_temperature_dict(Inputs):
    # Set the outlet temperature of the fluid for each heat exchanger in the Inputs dictionnary
    nb_hx = int(get_value('nb_hx', 'named_expression', Inputs))
    for i in range(1, nb_hx+1) :
        T_fluid_in = get_value(f'T_fluid_in_{i}', 'named_expression', Inputs)
        L = get_value(f'L_{i}', 'named_expression', Inputs)
        a_f = get_value(f'a_f_{i}', 'named_expression', Inputs)
        b_f = get_value(f'b_f_{i}', 'named_expression', Inputs)

        T_fluid_out_value = T_fluid_out(T_fluid_in, L, a_f, b_f)
        set_value(f'T_fluid_in_{i+1}', 'named_expression', Inputs, T_fluid_out_value)
    return Inputs