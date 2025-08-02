#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
from os import walk


# In[ ]:


# folders
spx_raw_folder_tick = "data/spx-raw/tick"
spx_raw_folder_eod = "data/spx-raw/eod/"
rf_raw_folder = "data/rf-raw/"

split_moneyness_folder_call = "data/split-moneyness/call"
split_moneyness_folder_put = "data/split-moneyness/put"

filter_folder_call = "data/filter/call"
filter_folder_put = "data/filter/put"
filter_folder = "data/filter"


# In[ ]:


# filenames
years = ["2022", "2023"]
frequency = ["spx_15x_", "spx_daily"]
quarters = ["q1_q2", "q3_q4"]

filenames_raw = {
    years[0]: {
        quarters[0]: frequency[0]+years[0]+quarters[0],  
        quarters[1]: frequency[0]+years[0]+quarters[1]},
            
    years[1]: {
        quarters[0]: frequency[0]+years[1]+quarters[0],  
        quarters[1]: frequency[0]+years[1]+quarters[1]}}


vol_skew_str ="_vol_skew"
filenames_vol_skew = {
    
    years[0]: {
        quarters[0]: frequency[1]+years[0]+quarters[0]+vol_skew_str,  
        quarters[1]: frequency[1]+years[0]+quarters[1]+vol_skew_str},
    years[1]: {
        quarters[0]: frequency[1]+years[1]+quarters[0]+vol_skew_str,  
        quarters[1]: frequency[1]+years[1]+quarters[1]+vol_skew_str}}

filename_call = "spx_daily_call"
filename_put = "spx_daily_put"
filename_call_imputed = "spx_daily_call_imputed"
filename_put_imputed = "spx_daily_put_imputed"

filenames_call_scaled = {
    "minmax": "spx_daily_call_minmax_scaled", 
    "standard": "spx_daily_call_standard_scaled", 
    "manual": "spx_daily_call_manual_scaled"
}

filenames_put_scaled = {
    "minmax": "spx_daily_put_minmax_scaled", 
    "standard": "spx_daily_put_standard_scaled", 
    "manual": "spx_daily_put_manual_scaled"
}

filenames_call_scaled = {
    "minmax": "spx_daily_call_minmax_scaled", 
    "standard": "spx_daily_call_standard_scaled", 
    "manual": "spx_daily_call_manual_scaled"
}

filenames_put_scaled = {
    "minmax": "spx_daily_put_minmax_scaled", 
    "standard": "spx_daily_put_standard_scaled", 
    "manual": "spx_daily_put_manual_scaled"
}


# In[ ]:


relevant_columns = ["[UNDERLYING_LAST]", "[STRIKE]", "[DTE]", "[RF]", "[REALIZED_VOL]", "[REALIZED_SKEW]", "[LAST]", "[QUOTE_UNIXTIME]", "[BID-ASK SPREAD]"]


# In[ ]:


def process_raw_tick_data(root, verbose=True):
    """
    Process raw SPX tick files obtained from Options DX.
    Parameters:
    root: string folder path root where the data is stored. 
    verbose: boolean variable for printing purposes
    """
    df = pd.DataFrame()
    flist = []
    for (dirpath, dirname, filename) in walk(root):
        if len(filename) !=0:
            flist.extend(filename)
            
    for f in flist:
        path = root  + f
        df_ = pd.read_table(path, delimiter = ",")
        if df.size == 0:
            df = df_
        else:
            df = pd.concat([df, df_])
        if verbose:
            print(f"Done with {f}")
     
    # fill blanks with nan
    df.columns = [column.strip() for column in df.columns]
    df.set_index(keys = "[QUOTE_DATE]", inplace = True)
    df = df.replace(" ", np.nan)
    df = df.astype("float", errors="ignore")
    
    return df


# In[ ]:


def process_raw_tbills_data(root, verbose=True):
    """
    Process raw US Tbills files as a proxy for risk free rate 
    from US treasuries website https://home.treasury.gov/
    Parameters:
    root: string folder path root where the data is stored. 
    verbose: boolean variable for printing purposes
    """
    root = rf_raw_folder
    rf = pd.DataFrame()
    flist = []
    for (dirpath, dirname, filename) in walk(root):
        if len(filename) !=0:
            flist.extend(filename)
            
    for f in flist:
        path = root + f
        df_ = pd.read_csv(path, index_col= "Date", parse_dates = True)
        if rf.size == 0:
            rf = df_
        else:
            rf = pd.concat([rf, df_])
        if verbose: 
            print(f"Done with {f}")
            
    return rf


# In[ ]:


def process_raw_eod_data(root, verbose=True):
    """
    Process raw SPX EOD files obtained from Options DX.
    Parameters:
    root: string folder path root where the data is stored. 
    verbose: boolean variable for printing purposes
    """
    df = pd.DataFrame()
    f = []
    d = []
    for (dirpath, dirname, filename) in walk(root):
        if len(filename) !=0:
            f.append(filename)
        d.extend(dirname)
        
    for (dirname, flist) in zip(d, f):
        for f in flist:
            path = root + dirname + "/" + f
            df_ = pd.read_table(path, delimiter = ",")
            if df.size == 0:
                df = df_
            else:
                df= pd.concat([df, df_])
            print(f"Done with {f}")
     
    # fill blanks with nan
    df.columns = [column.strip() for column in df.columns]
    df.set_index(keys = "[QUOTE_DATE]", inplace = True)
    df = df.replace(" ", np.nan)
    df = df.astype("float", errors="ignore")
    
    return df


# In[ ]:


def call_put_split(data):
    """
    Splits dataframe into 2 dataframes of calls and puts only, column-wise.
    Columns which have _C in name would be calls and _P will be puts.
    Returns two dataframes of calls and puts as tuple.
    """
    ccols = [c for c in data.columns if "C_" in c]
    pcols = [c for c in data.columns if "P_" in c]  

    data_call = data.drop(columns=pcols)
    data_put = data.drop(columns=ccols)
    
    columns =  [col.replace("C_", "") for col in data_call.columns]
    data_call.columns = columns
    data_put.columns = columns
    
    return data_call, data_put


# In[1]:


def clean_data(data, min_price=0.1, min_volume=0, split=False):
    """
    Filter SPX tick data based on the conditions described in *2023, Liu and Zhang*
    These conditions are regarding volume, and days to maturity and effectively reduce the size of the dataset.
    Returns a filtered dataframe.
    """   
    if split:
        data = data[data["[LAST]"] > min_price]
        data = data[data["[VOLUME]"] > min_volume]
    return data


# In[ ]:


def filter_data(data, columns=None):
    """
    Filter SPX tick data by columns, effectively leaving only relevant columns for training.
    Parameters:
    data - dataset
    columns - relevant columns.
    Returns a filtered dataset.
    """
    if columns:
        return data[columns]
    else:
        return data[relevant_columns]


# In[ ]:


def calculate_bid_ask_spread(data):
    data["[BID-ASK SPREAD]"] = data["[ASK]"] - data["[BID]"]
    return data

