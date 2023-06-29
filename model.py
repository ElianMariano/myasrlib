import tensorflow as tf
from tensorflow.keras.layers import LSTM, BatchNormalization, Dropout, Dense, MaxPooling1D, Flatten, Conv1D, Bidirectional, Conv2D, Reshape, TimeDistributed, Resizing,MaxPooling2D, Input, RandomFlip, RandomRotation
from tensorflow.keras.models import Sequential

def create_model(input_shape, output_len, n_conv, n_hidden, n_dense, batchnorm=True):
    model = Sequential()
    model.add(Conv2D(n_conv, kernel_size=(3, 3), activation='relu', input_shape=input_shape, padding='same', kernel_initializer='ones'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if (batchnorm):
        model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Flatten())
    for i in range(0, n_hidden):
        model.add(Dense(n_dense, activation='relu'))
    model.add(Dense(output_len, activation='softmax'))

    return model

if __name__ == '__main__':
    model = create_model((80, 80, 1), 62, 128, 7, 280)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    model.build()

    print(model.summary())