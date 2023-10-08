import os
import pandas as pd

import utils as u

METADATA_PATH = "./metadata.csv"

def main():

    metadata_df = pd.read_csv(METADATA_PATH)  

    try:
        u.generate_plot_from_multicol_metadata(metadata_df, chartType='line')
    except Exception as error:
        print("Stop generating Line graph")
        print("An exception occurred:", error)
    
    try:
        u.generate_plot_from_multicol_metadata(metadata_df, chartType='bar')
    except Exception as error:
        print("Stop generating Bar graph")
        print("An exception occurred:", error)
    
if __name__ == "__main__":
    main()