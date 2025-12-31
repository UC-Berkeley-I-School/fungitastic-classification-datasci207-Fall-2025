import pandas as pd
import os

from src.utils import load_yaml


def pre_process_metadata(cfg_path):
    config_ = load_yaml(cfg_path)
    
    take_fraction = config_['take_fraction']
    train_val_split = config_['train_val_split']
    
    img_cfg = config_['metadata']
    root_tab_path = img_cfg['datas_mini_path']

    root_tab_path = f"{os.getcwd()}{root_tab_path}"
    
    train_val_metadata = pd.read_csv(f"{root_tab_path}/FungiTastic-Mini-Train.csv")

    # Get total rows
    train_val_n = int(len(train_val_metadata) * take_fraction)
    # Get train only rows
    train_n = int(train_val_n * train_val_split)

    # select columns of interest and limit rows
    train_val_metadata = train_val_metadata[[
        'species','year','month','day','habitat','countryCode','hasCoordinate',
        'iucnRedListCategory','substrate','latitude','longitude','coorUncert',
        'region','district','metaSubstrate','poisonous','elevation','landcover',
        'biogeographicalRegion'
    ]]
    train_val_metadata = train_val_metadata.iloc[:train_val_n].reset_index(drop=True)

    # --- Robust missing-value handling ---
    # Fill categorical/string columns with a placeholder and numeric columns with median.
    for col in train_val_metadata.columns:
        if train_val_metadata[col].dtype == object or str(train_val_metadata[col].dtype).startswith('string'):
            train_val_metadata[col] = train_val_metadata[col].fillna('missing')
        else:
            # Coerce to numeric, compute median (ignore NaNs), fill with median
            train_val_metadata[col] = pd.to_numeric(train_val_metadata[col], errors='coerce')
            median = train_val_metadata[col].median()
            if pd.isna(median):
                median = 0
            train_val_metadata[col] = train_val_metadata[col].fillna(median)

    train_metadata = train_val_metadata.iloc[:train_n].reset_index(drop=True)
    val_metadata = train_val_metadata.iloc[train_n:].reset_index(drop=True)

    return train_metadata, val_metadata