import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
import warnings
import os
import seaborn as sns
from tqdm import tqdm
# from dataset.dataset_reader import read_file

# TODO Remove this function
def read_file(file_name, root_dir='') -> list:
    column=['path_from_data_dir_windows' if os.name == 'nt' else 'path_from_data_dir']

    data = pd.read_csv(os.path.join(root_dir, file_name), header=0).query('is_audio').filter(items=column).rename(columns={column[0]: 'path'})

    data = list(map(lambda x: x[0], data.values.tolist()))

    return list(filter(lambda x: x[-7:] == 'WAV.wav', data))

def read_label_classes(file_name='labels.csv') -> list:
    labels = pd.read_csv(file_name).set_index('phoneme_label').T.to_dict('list')

    return list(map(lambda x: x, labels))

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)

    mpl.rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    file_name = sys.argv[1]

    # Load file name

    train_files = read_file(file_name, root_dir='timit')

    label_files = list(map(lambda file: re.sub(r"(\.WAV)(\.wav)", '.PHN', file), train_files))

    # Load timestamps
    timestamps = np.array([])

    classes = read_label_classes()

    result = np.empty((0, len(classes)), dtype=np.int16)

    for file in tqdm(label_files):
        timestamp = pd.read_csv(os.path.join('timit', 'data', file), ' ', header=None).to_numpy()
        if len(timestamps) == 0:
            timestamps = pd.read_csv(os.path.join('timit', 'data', file), ' ', header=None).to_numpy()
        else:
            timestamps = np.concatenate((timestamps, timestamp), axis=0)
        
        timestamp = np.fromiter(map(lambda x: classes.index(x[2]), timestamp), dtype=np.int16)

        for i in range(0, len(timestamp)):
            column = np.zeros((1, len(classes),), dtype=np.int16)
            for j in range(0, len(classes)):
                column[0][j] = np.count_nonzero(timestamp == j)
            result = np.concatenate((result, column), axis=0)

    sum_df = pd.DataFrame(result, columns=classes).sum()
    
    labels = []
    for i in range(0, len(classes)):
        labels.append([int(i), int(sum_df[classes[i]]), classes[i]])
    
    labels_df = pd.DataFrame(labels, columns=['index', 'count', 'code'])
    # labels_df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), columns=['a', 'b', 'c'])

    # DataFrame
    # INDEX, CODE, COUNT, file
    #   1,     h#,    20,  ''
    sns.set_theme()
    sns.catplot(data=labels_df, x='index', y='count')
    plt.show()