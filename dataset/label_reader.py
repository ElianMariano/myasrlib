import numpy as np
import pandas as pd

def read_label(file_name, phoneme_labels) -> np.ndarray:
    sa1 = pd.read_csv(file_name, ' ', header=None).to_numpy()

    data = []
    for i in range(0, len(sa1)):
        data.append([sa1[i, 0], sa1[i, 1], phoneme_labels[sa1[i, 2]][0]])

    return np.array(data)
