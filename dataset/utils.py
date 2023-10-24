import os
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype

#################################################
# etc.
#################################################
def clean_title(value):
    return " ".join(value.split())

def check_folder_exsit(path:str):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Create... {path}")
    
    return True


#################################################
# Dataset Preparation
#################################################
def train_val_test_data(metadata_path:str, total_n:int, seed:int) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):

    metadata_df = pd.read_csv(metadata_path)
    line_bar_df = metadata_df.loc[(metadata_df["chartType"] == 'line') | (metadata_df["chartType"] == 'bar')]
    print(f"There are {len(line_bar_df)} data for line or bar chart.")

    line_bar_df = line_bar_df.sample(n=total_n, random_state=seed)
    bar_cnt = line_bar_df.groupby("chartType")["id"].count()[0]
    line_cnt = line_bar_df.groupby("chartType")["id"].count()[1]
    print(f"{total_n} Random sampling\nline -> {line_cnt}\tbar -> {bar_cnt}")

    train, test = train_test_split(line_bar_df, test_size=0.2, random_state=seed)
    train, val = train_test_split(train, test_size=0.125, random_state=seed)

    train_bar_cnt = train.groupby("chartType")["id"].count()[0]
    train_line_cnt = train.groupby("chartType")["id"].count()[1]
    val_bar_cnt = val.groupby("chartType")["id"].count()[0]
    val_line_cnt = val.groupby("chartType")["id"].count()[1]
    test_bar_cnt = test.groupby("chartType")["id"].count()[0]
    test_line_cnt = test.groupby("chartType")["id"].count()[1]

    print(f"Dataset split into train({len(train)}):val({len(val)}):test({len(test)})")
    print(f"Train : line -> {train_line_cnt}\tbar -> {train_bar_cnt}")
    print(f"Val : line -> {val_line_cnt}\tbar -> {val_bar_cnt}")
    print(f"Test : line -> {test_line_cnt}\tbar -> {test_bar_cnt}")

    return train, val, test



#################################################
# Save metadata
#################################################
def save_metadata(df:pd.DataFrame, path_to_save:str, update_file_name=False) -> pd.DataFrame:

    metadata = df[["imgPath", "first_caption", "chartType", "chartElement"]]
    metadata.iloc[:]["imgPath"] = metadata["imgPath"].map(lambda path: path.split("/")[-1])
    metadata.columns = ["file_name", "caption", "chartType", "chartElement"]
    if update_file_name:
        metadata.iloc[:]["file_name"] = metadata.iloc[:]['chartElement'].astype(str)+"_"+metadata.iloc[:]['file_name'].astype(str)
    metadata = metadata.reset_index(drop=True)
    metadata.head()

    check_folder_exsit(path_to_save)
    metadata.to_csv(os.path.join(path_to_save, 'metadata.csv'), index=False, header=True)
    print(f"Save metadata ... {os.path.join(path_to_save, 'metadata.csv')}")

    return metadata



#################################################
# Single Column Plotting Functions
#################################################
def base_plot(data:pd.DataFrame, metadata:pd.Series, plot_type:str, not_showing=True):

    ax = data.plot(
        x=data.columns[0], y=data.columns[1], # Default
        ylabel=data.columns[1],     # ~ AxisLabel
        kind=plot_type,
        grid=True,      # ~ Grid
        title=clean_title(metadata["title"]),     # ~ Title
        legend=False,       # ~ Legend(Only for multi column data)
        # xlabel=''     # ~ AxisLabel
        )
    
    fig = ax.get_figure()

    if not_showing:
        plt.close()
    return fig

def no_axislabel_plot(data:pd.DataFrame, metadata:pd.Series, plot_type:str, not_showing=True):
    
    ax = data.plot(
        x=data.columns[0], y=data.columns[1], # Default
        # ylabel=data.columns[1],     # ~ AxisLabel
        kind=plot_type,
        grid=True,      # ~ Grid
        title=clean_title(metadata["title"]),     # ~ Title
        legend=False,       # ~ Legend(Only for multi column data)
        xlabel=''     # ~ AxisLabel
        )
    
    fig = ax.get_figure()

    if not_showing:
        plt.close()
    return fig

def no_title_plot(data:pd.DataFrame, metadata:pd.Series, plot_type:str, not_showing=True):
    
    ax = data.plot(
        x=data.columns[0], y=data.columns[1], # Default
        ylabel=data.columns[1],     # ~ AxisLabel
        kind=plot_type,
        grid=True,      # ~ Grid
        # title=metadata["title"],     # ~ Title
        legend=False,       # ~ Legend(Only for multi column data)
        # xlabel=''     # ~ AxisLabel
        )
    
    fig = ax.get_figure()

    if not_showing:
        plt.close()
    return fig

def no_grid_plot(data:pd.DataFrame, metadata:pd.Series, plot_type:str, not_showing=True):
    
    ax = data.plot(
        x=data.columns[0], y=data.columns[1], # Default
        ylabel=data.columns[1],     # ~ AxisLabel
        kind=plot_type,
        # grid=True,      # ~ Grid
        title=clean_title(metadata["title"]),     # ~ Title
        legend=False,       # ~ Legend(Only for multi column data)
        # xlabel=''     # ~ AxisLabel
        )
    
    fig = ax.get_figure()

    if not_showing:
        plt.close()
    return fig

def generate_single_col_plot(df:pd.DataFrame, data_root:str, path_to_save:str, update_chart_name=False, is_verbose=False):

    check_folder_exsit(path_to_save)

    for i, item in tqdm(df.iterrows()):
        data_raw = item['dataPath'].split("/")[-1]

        data_name = data_raw.split(".")[0]
        chart_name = f"{data_name}.png"
        chart_element = item["chartElement"]
        chart_type = item["chartType"]
        
        data = pd.read_csv(os.path.join(data_root,data_raw))

        # Check the type of column value is numeric.
        if not (is_numeric_dtype(data[data.columns[1]])):
            old_dtype = data[data.columns[1]].dtype
            data[data.columns[1]] = data[data.columns[1]].str.replace("%","") # Delete Unit
            data[data.columns[1]] = pd.to_numeric(data[data.columns[1]],errors='coerce')
            new_dtype = data[data.columns[1]].dtype
            print(f"Not numeric column! Change {old_dtype} -> {new_dtype}")

        if update_chart_name:
            chart_name = f"{chart_element}_{chart_name}"
        chart_path = os.path.join(path_to_save, chart_name)
        if chart_element == "FullCover":
            fig = base_plot(data, item, plot_type=chart_type, not_showing=True)
            fig.savefig(chart_path,bbox_inches='tight')
        elif chart_element == "NoTitle":
            fig = no_title_plot(data, item, plot_type=chart_type, not_showing=True)
            fig.savefig(chart_path,bbox_inches='tight')
        elif chart_element == "NoAxisLabel":
            fig = no_axislabel_plot(data, item, plot_type=chart_type, not_showing=True)
            fig.savefig(chart_path,bbox_inches='tight')
        elif chart_element == "NoGrid":
            fig = no_grid_plot(data, item, plot_type=chart_type, not_showing=True)
            fig.savefig(chart_path,bbox_inches='tight')
        else:
            print(f"This chart element type '{chart_element}' is not covered!")

        if is_verbose:
            print(f"the {chart_element} plot by using {data_raw} is created at here ... {chart_path} !")



#################################################
# Multi Column Plotting Functions
#################################################
def base_multi_plot(data:pd.DataFrame, metadata:pd.Series, plot_type:str, not_showing=True):
    
    ax = data.plot(
        x=data.columns[0], #y=data.columns[1], # Default
        # ylabel=data.columns[1:],     # ~ AxisLabel
        kind=plot_type,
        grid=True,      # ~ Grid
        title=clean_title(metadata["title"]),     # ~ Title
        legend=True,       # ~ Legend(Only for multi column data)
        # xlabel=''     # ~ AxisLabel
        )
    plt.legend(loc='best')
    fig = ax.get_figure()

    if not_showing:
        plt.close()
    return fig

def multi_subplot(data:pd.DataFrame, metadata:pd.Series, plot_type:str, not_showing=True):
    
    ax = data.plot(
        x=data.columns[0], #y=data.columns[1], # Default
        #ylabel=data.columns[1],     # ~ AxisLabel
        kind=plot_type,
        grid=True,      # ~ Grid
        title=clean_title(metadata["title"]),     # ~ Title
        legend=True,       # ~ Legend(Only for multi column data)
        # xlabel=''     # ~ AxisLabel
        subplots=True
        )
    # print(ax)
    fig = ax[0].get_figure()

    if not_showing:
        plt.close()
    return fig

def generate_multi_col_plot(df:pd.DataFrame, data_root:str, path_to_save:str, update_chart_name=False, is_verbose=False):

    check_folder_exsit(path_to_save)

    for i, item in tqdm(df.iterrows()):
        data_raw = item['dataPath'].split("/")[-1]

        data_name = data_raw.split(".")[0]
        chart_name = f"{data_name}.png"
        chart_element = item["chartElement"]
        chart_type = item["chartType"]
        
        data = pd.read_csv(os.path.join(data_root,data_raw))

        # Check the type of column value is numeric.
        for col in data.columns[1:]:
            if not (is_numeric_dtype(data[col])):
                old_dtype = data[col].dtype
                data[col] = data[col].str.replace("%","") # Delete Unit
                data[col] = pd.to_numeric(data[col],errors='coerce')
                new_dtype = data[col].dtype
                print(f"Not numeric column! Change {old_dtype} -> {new_dtype}")
        
        if update_chart_name:
            chart_name = f"{chart_element}_{chart_name}"
        chart_path = os.path.join(path_to_save, chart_name)
        if chart_element == "OnePlot":
            fig = base_multi_plot(data, item, plot_type=chart_type, not_showing=True)
            fig.savefig(chart_path,bbox_inches='tight')
        elif chart_element == "SubPlot":
            fig = multi_subplot(data, item, plot_type=chart_type, not_showing=True)
            fig.savefig(chart_path,bbox_inches='tight')
        else:
            print(f"This chart element type '{chart_element}' is not covered!")

        if is_verbose:
            print(f"the {chart_element} plot by using {data_raw} is created at here ... {chart_path} !")
