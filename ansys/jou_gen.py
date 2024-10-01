import os 
import numpy as np
import pandas as pd

import shutil

import random
import string
import send2trash

computer = 'seagull'

if computer == 'seagull':
    fp_cmd = r"D:\ANSYS Fluent Projects\temp"
elif computer == 'lmps_cds':
    fp_cmd = "/usrtmp/delachaux/temp"

def generate_unlikely_combination(length=5):
    # Define a large set of characters (e.g., all letters, digits, and punctuation)
    char_pool = string.ascii_letters + string.digits + string.punctuation
    # Randomly sample from the character pool to create a string of the given length
    combination = ''.join(random.choices(char_pool, k=length))
    return combination
# Generate a combination of 5 characters

def create_folder(folder_path):
    """
    Creates a folder at the specified path.

    :param folder_path: Path where the folder will be created.
    """
    try:
        # Create the folder if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder created at {folder_path}")
        else:
            print(f"Folder already exists at {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def delete_folder(folder_path):
    """
    Deletes the folder at the specified path.

    :param folder_path: Path of the folder to be deleted.
    """
    try:
        # Delete the folder if it exists
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder deleted at {folder_path}")
        else:
            print(f"No folder found at {folder_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to replace placeholders
def replace_placeholders(line, df, row_index):
    if 'BC_ID' in line:
        line = line.replace('BC_ID', df.at[row_index, 'BC_ID'])
    if 'REPORT_NAME' in line:
        line = line.replace('REPORT_NAME', df.at[row_index, 'REPORT_NAME'])
    return line

# Reading the input file and writing to the output file
def process_file(input_file, output_file, df, row_index):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines[:]:
            modified_line = replace_placeholders(line, df, row_index)
            file.write(modified_line)

def concatenate_txt_files(folder_path, output_file):
    """
    Concatenates all .txt files in the specified folder and writes the result to the output file.
    
    :param folder_path: Path to the folder containing .txt files.
    :param output_file: Path to the output file where concatenated content will be saved.
    """
    all_texts = ["/file/set-tui-version \"22.2\""]

    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            
            # Read each file and append its content
            with open(file_path, 'r') as file:
                all_texts.append(file.read())

    # Concatenate all texts
    concatenated_text = "\n".join(all_texts)

    # Write the concatenated text to the output file
    with open(output_file, 'w') as file:
        file.write(concatenated_text)

def create_dataframe_from_complex_text_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    column_names = lines[0].strip().split('\t')
    data = {col: [] for col in column_names}

    for line in lines[1:]:
        values = line.strip().split('\t')
        
        # Ensure that the number of values matches the number of columns
        if len(values) == len(column_names):
            for col, value in zip(column_names, values):
                data[col].append(value)
        else:
            print(f"Warning: Line skipped due to mismatched number of columns: {line}")

    df = pd.DataFrame(data)
    return df

def insert_line_at_index(original_text, line_to_add, index):
    lines = original_text.split('\n')  # Split the original text into lines
    lines.insert(index, line_to_add)  # Insert the new line at the specified index
    modified_text = '\n'.join(lines)  # Re-join the lines back into a single string
    return modified_text

def generate_script(df, template, output_filename):
    with open(output_filename, 'w') as file:
        flag=0
        for index, row in df.iterrows():
            paragraph = template

            for var in ['VAR_coll_inlet_mdot', 'VAR_distrib_inlet_mdot', 'VAR_distrib_outlet_mdot', 'VAR_nb_it', 'VAR_pressure_report', 'VAR_mdot_report', 'VAR_residuals_report', 'VAR_datafile_name']:
                paragraph = paragraph.replace(var, str(row[var]))

            if index >= 1 and df.loc[index]['Q_max'] > df.loc[index-1]['Q_max']:
                paragraph = insert_line_at_index(paragraph, f"solve/initialize/hyb-initialization yes\n\n",4)
            else:
                pass

            if df.loc[index]['Vdot_max'] > 100 and flag == 0:
                flag = 1
                paragraph = insert_line_at_index(paragraph, f"/define/models/viscous/kw-sst yes\n\n",4)
            else:
                pass     

            file.write(paragraph + '\n\n')

# Function to parse the text content into a dictionary
def parse_text_to_dict(text):
    data_dict = {}
    current_key = ""
    for line in text.split("\n"):
        if line.startswith("(("):  # New section
            current_key = line.split('"')[1]  # Get label name
            data_dict[current_key] = []
        elif line.strip() and not line.startswith(")"):  # Data line
            _, value = line.strip().split("\t")
            data_dict[current_key].append(float(value))
    return data_dict

report_type_menu = {
    "Area": 0,
    "Integral": 1,
    "Standard Deviation": 2,
    "Flow Rate": 3,
    "Mass Flow Rate": 4,
    "Volume Flow Rate": 5,
    "Area-Weighted Average": 6,
    "Mass-Weighted Average": 7,
    "Sum": 8,
    "Uniformity Index - Mass Weighted": 9
}

report_field_variable_menu = {
    "Pressure": 0,
    "Density": 1,
    "Velocity": 2,
    "Properties": 3,
    "Wall Fluxes": 4,
    "Cell Info": 5,
    "Mesh": 6,
    "Residuals": 7,
    "Derivatives": 8
}

report_definition_menu = {
    "Pressure": {
                "Static Pressure": 0,
                "Pressure Coefficient": 1,
                "Dynamic Pressure": 2,
                "Absolute Pressure": 3,
                "Total Pressure": 4,
                "Relative Total Pressure": 5
                }
}

def tui_create_report_definitions(name,report_dic,surface_name_list):

    report_type = report_dic['Report Type']

    if report_type == "flux-massflow":
        surface_names = "zone-names"
    else:
        surface_names = "surface-names"

    field_variable = report_dic['Field Variable']

    surfaces_string = '\n'.join(surface_name_list) + '\n'

    template=f"""
solve/report-definitions/add {name}
{report_type}
field
{field_variable}
{surface_names}
""" + surfaces_string+ """
quit quit quit
"""
    
    return template

def read_residuals(file_path):
    with open(file_path, 'r') as file:
        text_content = file.read()

    data_dict = parse_text_to_dict(text_content)

    return pd.DataFrame(data_dict)

def write_report_gui(report_dic = {'Report Type': 1, 'Field Variable': 0, 'Definition': 0},surface_list=[0],file_path="test"):

    # 1 is for area weighted average
    # 0 for pressure
    # 0 for static pressure

    surface_list_string = " ".join([str(i) for i in surface_list])

    template = f"""
file/set-tui-version "22.2"
(cx-gui-do cx-activate-item "Ribbon*Frame1*Frame6(Results)*Table1*Table3(Reports)*PushButton4(Surface Integrals)")
(cx-gui-do cx-set-list-selections "Surface Integrals*Table1*DropDownList1(Report Type)" '( {report_dic['Report Type']}))
(cx-gui-do cx-activate-item "Surface Integrals*Table1*DropDownList1(Report Type)")
(cx-gui-do cx-set-list-selections "Surface Integrals*Table2*DropDownList1(Field Variable)" '( {report_dic['Field Variable']}))
(cx-gui-do cx-activate-item "Surface Integrals*Table2*DropDownList1(Field Variable)")
(cx-gui-do cx-set-list-selections "Surface Integrals*Table2*DropDownList2" '( {report_dic['Definition']}))
(cx-gui-do cx-activate-item "Surface Integrals*Table2*DropDownList2")
(cx-gui-do cx-set-list-selections "Surface Integrals*Table2*Table4*List1(Surfaces)" '( {surface_list_string}))
(cx-gui-do cx-activate-item "Surface Integrals*Table2*Table4*List1(Surfaces)")
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton4(Write)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "{file_path}") "Surface Report Files (*.srp)")
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton2(Cancel)")
    """
    return template

def tui_write_report_massflow(file_path):

    template = f"""
report/fluxes/mass-flow yes yes "{file_path}"
"""
    
    return template

def tui_write_report_ht(file_path):

    template = f"""
report/fluxes/heat-transfer yes yes "{file_path}"
"""
    
    return template

def tui_write_report_rad_ht(file_path):

    template = f"""
report/fluxes/rad-heat-trans yes yes "{file_path}"
"""
    
    return template

def gui_write_report_sp_prepared(file_path):

    template = f"""
/file/set-tui-version "22.2"
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton2(Cancel)")
(cx-gui-do cx-activate-item "Ribbon*Frame1*Frame6(Results)*Table1*Table3(Reports)*PushButton4(Surface Integrals)")
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton4(Write)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "{file_path}") "Surface Report Files (*.srp)") "Surface Report Files (*.srp)")
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton2(Cancel)")
"""
    return template

def change_bc_mdot_inlet(name,value):

    if type(value) == str:
        value = f'"{value}"'
    else:
        value = f'{value}'

    template = f"""
define/boundary-conditions/mass-flow-inlet {name} yes yes no {value} no 0 no yes
"""
    return template

def change_bc_velocity_inlet(name,value):

    if type(value) == str:
        value = f'"{value}"'
    else:
        value = f'{value}'

    template = f"""
define/boundary-conditions/ velocity-inlet {name} no no yes yes no {value} no 0
"""
    return template

def change_bc_pressure_outlet(name, value, backflow_direction_specification_method='Normal to Boundary',target_mass_flow_rate=False,mdot=0.,pressure_upper_bound = 5000000.,pressure_lower_bound = 1.):

    if type(value) == str:
        value = f'"{value}"'
    else:
        value = f'{value}'

    if type(mdot) == str:
        mdot = f'"{mdot}"'
    else:
        mdot = f'{mdot}'

    if backflow_direction_specification_method == 'Normal to Boundary':

        if target_mass_flow_rate == True:
            # raise error
            template = f"""
define/boundary-conditions/pressure-outlet {name} yes no {value} no yes yes no yes yes no {mdot} no {pressure_upper_bound} no {pressure_lower_bound}"""

        else: 
            template = f"""
define/boundary-conditions/pressure-outlet {name} yes no {value} no yes yes no no no"""
    
    elif backflow_direction_specification_method == 'From Neighboring Cell':

        if target_mass_flow_rate == False:
            template = f"""
define/boundary-conditions/pressure-outlet {name} yes no {value} no no yes yes no no no"""
        
        else:
            template = f"""
define/boundary-conditions/pressure-outlet {name} yes no {value} no no yes no no yes no {mdot} no 5000000 no 1"""
    
    elif backflow_direction_specification_method == 'prevent backflow':

        if target_mass_flow_rate == True:
            template = f"""
define/boundary-conditions pressure-outlet {name} no {value} no no yes no {mdot} no {pressure_upper_bound} no {pressure_lower_bound}"""
            
        else :
            template = f"""
define/boundary-conditions pressure-outlet {name} no {value} no no no"""

    else:
        raise Exception('Invalid backflow_direction_specification_method')
        
    return template

def save_journal(tui, string_list, file_name_wo_ext, read=True):
    
    concatenate_and_write_to_file(string_list, os.path.join(fp_cmd, f'{file_name_wo_ext}.txt'))

    if read:
        tui.file.read_journal(os.path.join(fp_cmd, f'{file_name_wo_ext}.txt'))
    else:
        pass

def change_named_expression(tui, named_expression, definition, unit):
    string_list = [f"""define/named-expressions/edit \"{named_expression}\" definition "{definition} [{unit}]" quit"""]
    save_journal(tui, string_list, 'change_named_expression')

def create_named_expression(tui, named_expression, definition, unit) :
    string_list = [f"""define/named-expressions/add \"{named_expression}\" definition "{definition} [{unit}]" quit"""]
    save_journal(tui, string_list, 'create_named_expression')

def change_bc_wall(tui, name, type, thickness, temperature):

    if type == "conductive_temperature_field":
        string_list = [f"""
    define/boundary-conditions/wall {name} {thickness} no 0 no yes temperature no {temperature} no no no 1"""]
    
    else:
        raise Exception('Invalid type')
    
    save_journal(tui, string_list, 'change_bc_wall')

def create_field(tui, named_expression, definition) :
    string_list = [f"""define/named-expressions/add \"{named_expression}\" definition "{definition}" quit"""]
    save_journal(tui, string_list, 'create_field')

def compute_surface_temperatures(tui, choice, fp):
    if choice == "pvt_slice_outdoor_Fluent_GMI_fins":
        string_list = [f"""(cx-gui-do cx-activate-item "Ribbon*Frame1*Frame6(Results)*Table1*Table3(Reports)*PushButton4(Surface Integrals)")
(cx-gui-do cx-set-list-selections "Surface Integrals*Table1*DropDownList1(Report Type)" '( 1))
(cx-gui-do cx-activate-item "Surface Integrals*Table1*DropDownList1(Report Type)")
(cx-gui-do cx-set-list-selections "Surface Integrals*Table2*DropDownList1(Field Variable)" '( 4))
(cx-gui-do cx-activate-item "Surface Integrals*Table2*DropDownList1(Field Variable)")
(cx-gui-do cx-activate-item "Surface Integrals*Table2*Table4*List1(Surfaces)")
(cx-gui-do cx-set-list-selections "Surface Integrals*Table2*Table4*List1(Surfaces)" '( 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104))
(cx-gui-do cx-activate-item "Surface Integrals*Table2*Table4*List1(Surfaces)")
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton4(Write)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "{fp}") "Surface Report Files (*.srp)")
(cx-gui-do cx-activate-item "Surface Integrals*PanelButtons*PushButton2(Cancel)")]"""]
        save_journal(tui, string_list, 'compute_surface_temperatures')
    else:
        raise Exception('Invalid choice')

def change_report_file_path(tui, report_name, chosen_path):

    string_list = [f"""solve/report-files/edit/{report_name} file-name \"{chosen_path}\""""]
    save_journal(tui, string_list, 'change_report_file_path')

def change_field(tui, named_expression, definition) :
    string_list = [f"""define/named-expressions/edit \"{named_expression}\" definition "{definition}" quit"""]
    save_journal(tui, string_list, 'change_existing_expression')

def change_gravity(tui, theta): # theta en rad
    #theta = np.deg2rad(theta)
    gravity_y = 9.81*np.sin(theta)
    gravity_z = 9.81*np.cos(theta)
    string_list = [f"""define/operating-conditions/gravity yes 0 {gravity_y} {gravity_z} quit"""]
    save_journal(tui, string_list, 'change_gravity')

def iterate(nb_it):
    return f"""solve/iterate {nb_it}
    """

def write_residuals_file(tui, folder_path, output_file_name_wo_ext):

    file_path = os.path.join(folder_path, f'{output_file_name_wo_ext}.txt')

    template = f"""
plot/residuals-set/plot-to-file "{file_path}" 
solve/iterate 1
plot/residuals-set/end-plot-to-file
"""
    string_list = [template]
    save_journal(tui, string_list, 'write_residuals_file')

def write_data(tui, folder_path, output_file_name_wo_ext):

    file_path = os.path.join(folder_path, f'{output_file_name_wo_ext}')

    template = f"""file/write-data "{file_path}"
    """

    string_list = [template]
    save_journal(tui, string_list, 'write_data')

def write_case(tui, folder_path, output_file_name_wo_ext):

    file_path = os.path.join(folder_path, f'{output_file_name_wo_ext}.txt')

    template = f"""file/write-case "{file_path}"
    """

    string_list = [template]
    save_journal(tui, string_list, 'write_case')

def standard_initialization(tui, surface_index, gauge_pressure_value, x_velocity_value, y_velocity_value, z_velocity_value):
    string_list = [f"""/file/set-tui-version "22.2"
(cx-gui-do cx-activate-item "Ribbon*Frame1*Frame5(Solution)*Table1*Table3(Initialization)*ButtonBox1(Method)*PushButton4(Options)")
(cx-gui-do cx-set-list-selections "Solution Initialization*Table1*DropDownList1(Compute from)" '( {surface_index}))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*DropDownList1(Compute from)")
(cx-gui-do cx-set-real-entry-list "Solution Initialization*Table1*Table7(Initial Values)*RealEntry1(Gauge Pressure)" '( {gauge_pressure_value}))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Table7(Initial Values)*RealEntry1(Gauge Pressure)")
(cx-gui-do cx-set-real-entry-list "Solution Initialization*Table1*Table7(Initial Values)*RealEntry2(X Velocity)" '( {x_velocity_value}))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Table7(Initial Values)*RealEntry2(X Velocity)")
(cx-gui-do cx-set-real-entry-list "Solution Initialization*Table1*Table7(Initial Values)*RealEntry3(Y Velocity)" '( {y_velocity_value}))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Table7(Initial Values)*RealEntry3(Y Velocity)")
(cx-gui-do cx-set-real-entry-list "Solution Initialization*Table1*Table7(Initial Values)*RealEntry4(Z Velocity)" '( {z_velocity_value}))
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Table7(Initial Values)*RealEntry4(Z Velocity)")
(cx-gui-do cx-activate-item "Solution Initialization*Table1*Frame9*PushButton1(Initialize)")"""]

    save_journal(tui, string_list, 'standard_initialization')

def write_time(tui, folder_path, output_file_name_wo_ext):

    file_path = os.path.join(folder_path, f'{output_file_name_wo_ext}.txt')

    template = f"""
file/start-transcript "{file_path}"
parallel/timer/usage
parallel/timer/reset
file/stop-transcript
"""
    string_list = [template]
    save_journal(tui, string_list, 'write_time')

def hybrid_initialization():
    return f"""
solve/initialize/hyb-initialization yes\n\n
"""

def define_viscous_kw_sst():
    return f"""
define/models/viscous/kw-sst yes\n\n
"""

def concatenate_and_write_to_file(strings, file_path):
    # Concatenate the strings with line breaks
    content = "\n".join(strings)
    
    # Open the file at the given path in write mode, overwriting if it exists
    with open(file_path, 'w') as file:
        file.write(content)

def convert_residuals_csv(folder_path,liste):
    df1 = pd.DataFrame()
    for i in liste:
        file_path = os.path.join(folder_path, f'DP{i}_residuals_report.txt')
        df2 = read_residuals(file_path)
        
        unique = pd.concat([df2, df1]).drop_duplicates(keep=False)
        unique.reset_index(drop=True,inplace=True)
        unique.to_csv(os.path.join(folder_path, f'DP{i}_residuals_report.csv'),index=True)
        df1 = pd.concat([df1,df2])

        # Check if file exists before trying to delete it
        if os.path.exists(file_path):
            if computer == 'seagull':
                send2trash.send2trash(file_path)
            else:
                os.remove(file_path)

def parse_report_to_dataframe(file_path,column_name):

    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file]

    first_line = None

    for i,line in enumerate(lines):
        if line.startswith('---'):
            start_index = i+1
            break

    lines = lines[start_index:]

    for i,line in enumerate(lines):
        if line.startswith('---'):
            stop_index = i
            break

    lines = lines[:stop_index]

    names = []
    values = []

    for item in lines:
        # Splitting each string by its rightmost whitespace
        name, value = item.rsplit(maxsplit=1)
        names.append(name)
        values.append(float(value))

    # Creating the DataFrame
    df = pd.DataFrame({'Component': names, column_name: values})

    return df

def convert_report(folder_path,file_name,column_name,output_folder_path,output_file_name_wo_extension=''):

    if output_file_name_wo_extension == '':
        output_file_name_wo_extension = file_name
    else:
        pass

    file_path = os.path.join(folder_path, file_name)

    parse_report_to_dataframe(file_path,column_name).to_csv(os.path.join(output_folder_path, f'{output_file_name_wo_extension}.csv'),index=False)

    # Check if file exists before trying to delete it
    if os.path.exists(file_path):
        if computer == 'seagull':
            send2trash.send2trash(file_path)
        else:
            os.remove(file_path)

def convert_parametric_reports(folder_path, report_type, liste):
    for i in liste:
        convert_report(folder_path,f'DP{i}_{report_type}',report_type)

def change_bc_type(tui, name, type):
    string_list = [f"""define/boundary-conditions/zone-type {name} {type}"""]
    save_journal(tui, string_list, 'change_bc_type')

def write_report(tui,measure,output_folder_path,output_file_name_wo_ext):

    if measure == 'mdot':
        with open(os.path.join(fp_cmd,'cmd_temp.txt'), "w") as file:
            file.write(tui_write_report_massflow(os.path.join(fp_cmd, f"output_temp_{measure}.txt")))

    elif measure == 'sp':
        with open(os.path.join(fp_cmd,'cmd_temp.txt'), "w") as file:
            file.write(gui_write_report_sp_prepared(os.path.join(fp_cmd, f"output_temp_{measure}.txt")))

    elif measure == 'ht':
        with open(os.path.join(fp_cmd,'cmd_temp.txt'), "w") as file:
            file.write(tui_write_report_ht(os.path.join(fp_cmd, f"output_temp_{measure}.txt")))

    elif measure == 'rad_ht':
        with open(os.path.join(fp_cmd,'cmd_temp.txt'), "w") as file:
            file.write(tui_write_report_rad_ht(os.path.join(fp_cmd, f"output_temp_{measure}.txt")))

    else:
        raise Exception('Invalid measure name')

    tui.file.read_journal(os.path.join(fp_cmd, "cmd_temp.txt"))

    convert_report(folder_path=fp_cmd,file_name=f"output_temp_{measure}.txt",column_name=measure,output_folder_path=output_folder_path,output_file_name_wo_extension=f'{output_file_name_wo_ext}')

def export(folder_path,name):

    df = pd.read_csv(os.path.join(folder_path, f'mdot_report_{name}.csv'))
    df2 = pd.read_csv(os.path.join(folder_path, f'sp_report_{name}.csv'))

    merged_df = pd.merge(df, df2, on='Component')
    dff = merged_df[merged_df['Component'].str.contains('coll_ch|distrib_ch')]
    dff['Component'].astype(str)

    coll_df = dff[dff['Component'].str.contains('coll_ch')].copy()
    coll_df['index'] = coll_df['Component'].str.extract(r'coll_ch_(\d+)')
    coll_df.dropna(subset='index', inplace=True)
    coll_df['index'] = coll_df['index'].astype(int)
    coll_df.sort_values(by='index', inplace=True)
    coll_df.set_index('index', inplace=True, drop=True)

    distrib_df = dff[dff['Component'].str.contains('distrib_ch')].copy()
    distrib_df['index'] = distrib_df['Component'].str.extract(r'distrib_ch_(\d+)')
    distrib_df.dropna(subset='index', inplace=True)
    distrib_df['index'] = distrib_df['index'].astype(int)
    distrib_df.sort_values(by='index', inplace=True)
    distrib_df.set_index('index', inplace=True, drop=True)

    distrib_df['mdot'] *= -1

    return distrib_df, coll_df

def write_plot_xy(tui, name, output_folder_path, output_file_name_wo_ext):

    file_path = os.path.join(output_folder_path, f'{output_file_name_wo_ext}.csv')

    template = f"""
/file/set-tui-version "22.2"
(cx-gui-do cx-set-list-tree-selections "NavigationPane*Frame2*Table1*List_Tree2" (list "Results|Plots|XY Plot|{name}"))
(cx-gui-do cx-set-toggle-button2 "Solution XY Plot*Table1*Table1*ButtonBox1(Options)*CheckButton4(Write to File)" #t)
(cx-gui-do cx-activate-item "Solution XY Plot*Table1*Table1*ButtonBox1(Options)*CheckButton4(Write to File)")
(cx-gui-do cx-activate-item "Solution XY Plot*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "{file_path}") "XY Files (*.xy)")"""
    
    string_list = [template]
    save_journal(tui, string_list, 'write_plot_xy')

def change_mesh(tui, mesh_path, mesh_name_wo_ext):
    file_path = os.path.join(mesh_path, f'{mesh_name_wo_ext}.msh.h5')
    string_list = [f"""/file/set-tui-version "22.2"
(cx-gui-do cx-activate-item "MenuBar*ReadSubMenu*Mesh...")
(cx-gui-do cx-set-toggle-button2 "Read Mesh Options*Table1(Options)*ToggleBox1*Replace mesh" #t)
(cx-gui-do cx-activate-item "Read Mesh Options*Table1(Options)*ToggleBox1*Replace mesh")
(cx-gui-do cx-activate-item "Read Mesh Options*PanelButtons*PushButton1(OK)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "{file_path}") "CFF Mesh Files (*.msh.h5 )")"""]

    save_journal(tui, string_list, 'change_mesh')

def create_radiation(tui, S2S_fp, caoMeshCode): 
    string_list = [f"""/file/set-tui-version "22.2"
(cx-gui-do cx-activate-item "Ribbon*Frame1*Frame3(Physics)*Table1*Table3(Models)*PushButton2(Radiation)")
(cx-gui-do cx-activate-item "Radiation Model*Table1*Frame3*Table1*Table2(View Factors and Clustering)*PushButton1( Compute/Write/Read)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "{os.path.join(S2S_fp,caoMeshCode)}.s2s.h5") "CFF S2S Files (*.s2s.h5 )")
(cx-gui-do cx-activate-item "Radiation Model*PanelButtons*PushButton1(OK)")"""]
    save_journal(tui, string_list, 'create_radiation')

def read_radiation(tui, S2S_fp, caoMeshCode) :
    string_list = [f"""/file/set-tui-version "22.2"
(cx-gui-do cx-activate-item "Ribbon*Frame1*Frame3(Physics)*Table1*Table3(Models)*PushButton2(Radiation)")
(cx-gui-do cx-activate-item "Radiation Model*Table1*Frame3*Table1*Table2(View Factors and Clustering)*PushButton2(   Read Existing File)")
(cx-gui-do cx-set-file-dialog-entries "Select File" '( "{os.path.join(S2S_fp,caoMeshCode)}.s2s.h5") "CFF S2S Files (*.s2s.h5 )")
(cx-gui-do cx-activate-item "Radiation Model*PanelButtons*PushButton1(OK)")"""]
    save_journal(tui, string_list, 'read_radiation')

def compute_mass_flow_rate(tui, surface_name, output_folder_path, output_file_name_wo_ext):
    file_path = os.path.join(output_folder_path, output_file_name_wo_ext)
    string_list = [f"""report/surface-integrals/mass-flow-rate {surface_name} () yes \"{file_path}\" quit"""]
    save_journal(tui, string_list, 'compute_mass_flow_rate')

def compute_temp_avg(tui, surface_name, output_folder_path, output_file_name_wo_ext):
    file_path = os.path.join(output_folder_path, output_file_name_wo_ext)
    string_list = [f"""report/surface-integrals/facet-avg {surface_name} () temperature yes \"{file_path}\" quit"""]
    save_journal(tui, string_list, 'compute_temp_avg')