# Simple_Tensorflow_sequentialdata
Simple Tensorflow for sequential data learning and prediction

<p align="center" width="100%">
    <img width="60%" src="https://github.com/jkaewprateep/Simple_Tensorflow_sequentialdata/blob/main/tensorflow_01.png">
    <img width="12%" src="https://github.com/jkaewprateep/Simple_Tensorflow_sequentialdata/blob/main/kid_40.png">
    <img width="24%" src="https://github.com/jkaewprateep/Simple_Tensorflow_sequentialdata/blob/main/kid_39.png"> </br>
    <b> Learning TensorFlow SDK and Python codes </b> </br>
</p>
</br>
</br>

## Predictions
<p align="center" width="100%">
    <img width="99%" src="https://github.com/jkaewprateep/Simple_Tensorflow_sequentialdata/blob/main/prediction_01.png"> </br>
    <img width="80%" src="https://github.com/jkaewprateep/Simple_Tensorflow_sequentialdata/blob/main/prediction_02.png"> </br>
    <b> Predictions </b> </br>
</p>
</br>
</br>

## Libraries import

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Libraries import
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# import root libraries
import os
import os.path
import time

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone

# Print Python version
print("TF version: ", tf.__version__);
```

## Variables

```
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
## variables
target_path = "D:\\Backup\\DataSet\\TQQQ\\TQQQ\\";
target_loggingpath = "D:\\Backup\\Logging\\logfile_20241025-23.log";
model_savedfile = "D:\\Backup\\Model\\mytensorflow_20241025-23.h5";
merged_file = "D:\\Backup\\MergedFile\\merged_file-2.csv";
merged_recordscount = 0;

## output variables
global new_df;
global time_interval;
global min_datetimestamp;
global max_datetimestamp;
global stop_reason;
global time_begin;
global time_elapse;

# 1 - ffill, else drop na
fill_option = 1;
time_interval = 15; # in seconds;
patience = 25;
select_recordthereshold = 0.8;
select_recordlimit = 20000;
num_trainingepoach = 10000;
optimizer_learningrates = 0.0001;
optimizer_momentum = 0.8;
num_stepperepoach = 1;
min_datetimestamp = None;
max_datetimestamp = None;
time_begin = None;
timezone = timezone(timedelta(hours=-4));
stop_reason = 0;

mergedcolumnnames_array = ["timestamp", "order", "open_x", "high_x", "low_x", "close_x", "volume_x", "vwap_x", "open_y",
    "high_y", "low_y", "close_y", "volume_y", "vwap_y", "increase"];

# add new column #increase for labels input.
new_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "vwap", "increase"]);
```

## Read dataset from .csv format file

```
def read_folder( directory_path ):
    ### define variable
    listof_directory = [];

    ### for each file in directory
    for _obj in os.listdir(target_path) :
        listof_directory.append(_obj);
    
    ### return list of directories
    return listof_directory;
    
def read_csv( file_name ):
    _filename = os.path.join( target_path , file_name );
    df = pd.read_csv( _filename );
    
    # return dataset
    return df;

def compare_openhigh_value( row ):
    if float(row["open"]) < float(row["high"]):
        val = 1;
    elif float(row["open"]) > float(row["high"]):
        val = 0;
    else:
        val = 0;
    return val
    
def convert_timestamp_value( row ):
    global previous_datetimestamp; 
    global time_interval;
    global min_datetimestamp;
    global max_datetimestamp;
    
    # aside to create new timestamp dataset
    if min_datetimestamp == None :
        min_datetimestamp = str(row["timestamp"]);
    elif datetime.fromisoformat(min_datetimestamp) > datetime.fromisoformat(str(row["timestamp"])) :
        min_datetimestamp = str(row["timestamp"]);
        
    if max_datetimestamp == None :
        max_datetimestamp = str(row["timestamp"]);
    elif datetime.fromisoformat(max_datetimestamp) < datetime.fromisoformat(str(row["timestamp"])) :
        max_datetimestamp = str(row["timestamp"]);

    return datetime.fromisoformat( str(row["timestamp"]) ).timestamp();
    
def read_dataset_fromfolder( _target_path ):
    # variable
    global new_df;

    # read dataset from folder
    _listof_directory = read_folder( _target_path );
    
    # list file object in folder
    for _obj in _listof_directory :
        ### read and display file name
        print( os.linesep );
        print( _obj )
        _df = read_csv( _obj );
        _df = pd.DataFrame( _df );
        
        _df["timestamp"] = _df["timestamp"].astype(str);
        
        print(_df.head())
        
        _df["timestamp"] = pd.to_datetime( _df["timestamp"] );              
        
        ### cleaning dataset
        _df['increase'] = _df.apply(compare_openhigh_value, axis=1);
        
        ### data types
        _df["timestamp"] = _df.apply(convert_timestamp_value, axis=1) / 1000000;        
        _df['open'] = _df['open'].astype(np.float32);
        _df['high'] = _df['high'].astype(np.float32);
        _df['low'] = _df['low'].astype(np.float32);
        _df['close'] = _df['close'].astype(np.float32);
        _df['volume'] = _df['volume'].astype(np.float32) / 100;
        _df['vwap'] = _df['vwap'].astype(np.float32);
        _df['increase'] = _df['increase'].astype(np.float32);

        print( _df.head(10) )        
        
        # concatenate for new dataset
        new_df = pd.concat([new_df, _df], axis=0, ignore_index=True);

    return new_df;

def compare_twocolumns_open_nonan( row ):
    if row["open_x"] == np.nan or row["open_x"] == 0 or str(row["open_x"]).upper() == "nan".upper() :
        return row["open_y"];
        
    return row["open_x"];
```
## Option - fill time interval from export data for create periods accumulation
```
def fill_timebetweeninterval( _obj ):

    new_datetime = datetime.fromisoformat( _obj ) + timedelta(seconds=time_interval);
    new_datetime = datetime(new_datetime.year, new_datetime.month, new_datetime.day, new_datetime.hour,
                        new_datetime.minute, new_datetime.second, tzinfo=timezone).isoformat();
    new_datetime = new_datetime.replace("T", " ");

    return new_datetime;


def fill_datarecordby_interval( _min_datetimestamp, _max_datetimestamp ):
    global time_interval;
    global timezone;
    global new_datetimestamp_array;
    
    target_number = datetime.fromisoformat( _max_datetimestamp ) - datetime.fromisoformat( _min_datetimestamp );
    target_number = target_number.seconds;

    # create new interval seconds array
    start = 0;
    stop = target_number + time_interval;
    step = time_interval;
    new_arrayseconds = np.arange(start, stop, step);
    
    # create new array of datetimestamp from interval
    new_arraytimeinterval = [];

    for item in new_arrayseconds :
        new_datetime = datetime.fromisoformat( _min_datetimestamp ) + timedelta(seconds=int(item));
        new_datetime = datetime(new_datetime.year, new_datetime.month, new_datetime.day, new_datetime.hour,
                        new_datetime.minute, new_datetime.second, tzinfo=timezone).isoformat();
        new_datetime = new_datetime.replace("T", " ");
        new_arraytimeinterval.append(new_datetime);

    return new_arraytimeinterval;
```








---

<p align="center" width="100%">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset.png">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset_2.png"> </br>
    <b> 🥺💬 รับจ้างเขียน functions </b> </br>
</p>
