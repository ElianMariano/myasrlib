import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense, MaxPooling1D, Flatten, Conv1D, Bidirectional, Conv2D, Reshape, TimeDistributed, Resizing,MaxPooling2D, Input
from tensorflow.keras.models import Sequential

def create_model(output_len, input_shape):
    model = Sequential()
    model.add(Input(shape=(input_shape[0], input_shape[1])))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(output_len, activation='relu'))

    return model

if __name__ == '__main__':
    model = create_model(1, (128, 161))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0625)

    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    print(model.summary())