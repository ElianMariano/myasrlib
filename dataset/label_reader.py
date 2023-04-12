import numpy as np
import pandas as pd
import tensorflow as tf

def read_label_classes(file_name='labels.csv') -> list:
    labels = pd.read_csv(file_name).set_index('phoneme_label').T.to_dict('list')

    return list(map(lambda x: x, labels))

def read_label(file_name, classes) -> tf.Tensor:
    sa1 = pd.read_csv(file_name, ' ', header=None).to_numpy()

    data = []

    for i in range(0, len(sa1)):
      prob = np.zeros(len(classes), dtype=np.int16)
      prob[classes.index(sa1[i, 2])] = 1
      prob = np.reshape(prob, (1, prob.shape[0]))

      if len(data) == 0:
        data = prob
      else:
        data = np.concatenate([data, prob], axis=0)
      
      # data.append([sa1[i, 0], sa1[i, 1], classes[sa1[i, 2]][0]])

    return tf.convert_to_tensor(data)