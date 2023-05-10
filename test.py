import os
import numpy as np
import warnings
import tensorflow as tf
from tensorflow.python.keras.callbacks import CSVLogger
from dataset.read_audio import read_audio_dataset, read_dataset_with_frames
from dataset.dataset_reader import read_file
from model import create_model
from config import read_config
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_cm(labels, predictions, p=0.5):
  cm = confusion_matrix(labels, predictions > p)
  plt.figure(figsize=(5,5))
  sns.heatmap(cm, annot=True, fmt="d")
  plt.title('Confusion matrix @{:.2f}'.format(p))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

if __name__ == '__main__':
    # Ignores the warnings thrown by pandas library
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)

    config = read_config()

    SPARSE = True if config['sparse'] == 'true' else False

    model = tf.keras.models.load_model('save_model/model')

    files = read_file('test_data_sm.csv', root_dir='timit')

    (x, y) = read_dataset_with_frames(files, root_dir=os.path.join('timit', 'data'), sparse=SPARSE)

    # val_x = x[0][tf.newaxis, ...]
    # val_y = y[0][tf.newaxis, ...]

    baseline_results = model.evaluate(x, y, batch_size=4)

    test_predictions_baseline = model.predict(x[0][tf.newaxis, ...], batch_size=8)

    # for name, value in zip(model.metrics_names, baseline_results):
    #     print(name, ': ', value)
    #     print()

    #     plot_cm(y[0][tf.newaxis, ...], test_predictions_baseline)