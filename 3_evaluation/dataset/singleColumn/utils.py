import os
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def check_folder_exsit(path:str):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print(f"Create... {path}")
    
    return True

def metadata_path_parser(path:str) -> str:
    path_comp = os.path.normpath(path).split(os.sep)
    new_path = os.path.join("./",path_comp[-2], path_comp[-1])
    return new_path

def base_plot(data:pd.DataFrame, metadata:pd.Series, plot_type:str, not_showing=True):

    ax = data.plot(
        x=data.columns[0], y=data.columns[1], # Default
        ylabel=data.columns[1],     # ~ AxisLabel
        kind=plot_type,
        grid=True,      # ~ Grid
        title=metadata["title"],     # ~ Title
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
        title=metadata["title"],     # ~ Title
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
        title=metadata["title"],     # ~ Title
        legend=False,       # ~ Legend(Only for multi column data)
        # xlabel=''     # ~ AxisLabel
        )
    
    fig = ax.get_figure()

    if not_showing:
        plt.close()
    return fig

def generate_plot_from_metadata(metadata:pd.DataFrame, chartType:str, n_plots=100, random_seed=42):

    # Check whether the save folder is already exist. If not, create one.
    check_folder_exsit(f"./eval_base/{chartType}/test")
    check_folder_exsit(f"./eval_no_axis_label/{chartType}/test")
    check_folder_exsit(f"./eval_no_title/{chartType}/test")
    check_folder_exsit(f"./eval_no_grid/{chartType}/test")

    # Load metadata and clean it
    metadata_df = metadata.loc[metadata["chartType"]==chartType]
    metadata_df = metadata_df.sample(n=n_plots, random_state=random_seed).sort_index()
    metadata_df = metadata_df[["dataPath", "title", "first_caption"]]
    metadata_df['dataPath'] = metadata_df['dataPath'].map(lambda x:metadata_path_parser(x))
    metadata_df['title'] = metadata_df['title'].map(lambda x:x.strip())
    print("\n\n\nMetadata information : ")
    print(metadata_df.info())
    print(metadata_df)
    print("\n\n\n")
    # Create 4 type plot with each raw data.
    new_metadata = {"file_name":[],"caption":[]}
    for i, metadata in tqdm(metadata_df.iterrows()):
        data_path = metadata['dataPath']
        data_name = os.path.split(data_path)[-1].split(".")[0]
        data_caption = metadata['first_caption']
        plot_name = f"{chartType}_{data_name}.png"
        
        data = pd.read_csv(data_path)
        print(f"Start with {data_path}")
        if not (is_numeric_dtype(data[data.columns[1]])):
            old_dtype = data[data.columns[1]].dtype
            data[data.columns[1]] = data[data.columns[1]].str.replace("%","") # Delete Unit
            data[data.columns[1]] = pd.to_numeric(data[data.columns[1]],errors='coerce')
            new_dtype = data[data.columns[1]].dtype
            print(f"Not numeric column! Change {old_dtype} -> {new_dtype}")
            
        fig1 = base_plot(data, metadata, plot_type=chartType, not_showing=True)
        fig2 = no_axislabel_plot(data, metadata, plot_type=chartType, not_showing=True)
        fig3 = no_title_plot(data, metadata, plot_type=chartType, not_showing=True)
        fig4 = no_grid_plot(data, metadata, plot_type=chartType, not_showing=True)

        fig1.savefig(f'./eval_base/{chartType}/test/{plot_name}',bbox_inches='tight')
        fig2.savefig(f'./eval_no_axis_label/{chartType}/test/{plot_name}',bbox_inches='tight')
        fig3.savefig(f'./eval_no_title/{chartType}/test/{plot_name}',bbox_inches='tight')
        fig4.savefig(f'./eval_no_grid/{chartType}/test/{plot_name}',bbox_inches='tight')

        new_metadata["file_name"] = new_metadata["file_name"] + ["./test/"+plot_name]
        new_metadata["caption"] = new_metadata["caption"] + [data_caption]

        print(f"Plot Generated: ./[eval_base | eval_no_axis_label | eval_no_title | eval_no_grid]/{chartType}_{data_name}.png")
        print()
    # Save the metadata for generated plots
    new_metadata_df = pd.DataFrame.from_dict(new_metadata) 
    new_metadata_df.to_csv (f'./eval_base/{chartType}/metadata.csv', index=False, header=True)
    new_metadata_df.to_csv (f'./eval_no_axis_label/{chartType}/metadata.csv', index=False, header=True)
    new_metadata_df.to_csv (f'./eval_no_title/{chartType}/metadata.csv', index=False, header=True)
    new_metadata_df.to_csv (f'./eval_no_grid/{chartType}/metadata.csv', index=False, header=True)

    print("\n\n\n")
    print(new_metadata_df.info())
    print(new_metadata_df.head())

    return True
