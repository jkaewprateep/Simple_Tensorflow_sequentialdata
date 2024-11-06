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
    <img width="80%" src="https://github.com/jkaewprateep/Simple_Tensorflow_sequentialdata/blob/main/prediction.png"> </br>
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











---

<p align="center" width="100%">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset.png">
    <img width="30%" src="https://github.com/jkaewprateep/advanced_mysql_topics_notes/blob/main/custom_dataset_2.png"> </br>
    <b> 🥺💬 รับจ้างเขียน functions </b> </br>
</p>
