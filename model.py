import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense, MaxPooling1D, Flatten, Conv1D, Bidirectional, Conv2D, Reshape, TimeDistributed, Resizing,MaxPooling2D, Input, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential

def create_model(input_shape, output_len):
    model = Sequential()
    # Data augmentation
    # model.add(RandomFlip("horizontal_and_vertical"))
    # model.add(RandomRotation(0.2))
    model.add(Conv2D(1, 6, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Conv1D(32, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu',))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(300, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu',))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # model.add(Dense(output_len, activation='relu'))
    model.add(Dense(output_len, activation='softmax'))

    return model

if __name__ == '__main__':
    model = create_model((60, 60), 62)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0625)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.build()

    print(model.summary())