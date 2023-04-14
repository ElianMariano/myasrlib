import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense, MaxPooling1D, Flatten, Conv1D, Bidirectional, Conv2D, Reshape, TimeDistributed, Resizing,MaxPooling2D, Input
from tensorflow.keras.models import Sequential

def create_model(input_shape, output_len):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1800, activation='relu'))
    model.add(Dense(1700, activation='relu'))
    model.add(Dense(1400, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(250, activation='relu'))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_len, activation='relu'))

    return model

if __name__ == '__main__':
    model = create_model((128, 61), 62)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0625)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    print(model.summary())