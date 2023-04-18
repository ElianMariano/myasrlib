import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    file_name = sys.argv[1]

    data = pd.read_csv(file_name, ',', header=None).to_numpy()

    accuracy = np.fromiter(map(lambda x: x[1], data[1:]), dtype=np.float32)

    loss = np.fromiter(map(lambda x: x[2], data[1:]), dtype=np.float32)

    val_accuracy = np.fromiter(map(lambda x: x[3], data[1:]), dtype=np.float32)

    val_loss = np.fromiter(map(lambda x: x[4], data[1:]), dtype=np.float32)

    x = np.arange(0, len(accuracy), 1, dtype=np.int16)

    # for i in range(0, len(x)):
    #     print(str(i) + ': ' + str(x[i]) + ', ' + str(accuracy[i]))

    if len(sys.argv) > 2:
        plt.title(sys.argv[2])

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.plot(x, accuracy, label='accuracy')
    plt.plot(x, loss, label='loss')
    plt.plot(x, val_accuracy, label='val_accuracy')
    plt.plot(x, val_loss, label='val_loss')
    plt.legend()
    plt.show()