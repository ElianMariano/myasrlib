"""
This file only reads the name of the .csv file for training or testing.
"""

import pandas as pd
import os

def read_file(file_name) -> list:
    column=['path_from_data_dir_windows' if os.name == 'nt' else 'path_from_data_dir']

    data = pd.read_csv(file_name, header=0).query('is_audio').filter(items=column).rename(columns={column[0]: 'path'})

    data = list(map(lambda x: x[0], data.values.tolist()))

    return list(filter(lambda x: x[-7:] == 'WAV.wav', data))

def read_labels(file_name='labels.csv') -> pd.DataFrame:
    return pd.read_csv(file_name).set_index('phoneme_label').T.to_dict('list')