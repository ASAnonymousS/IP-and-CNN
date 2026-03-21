import tensorflow as tf
import tensorflow.keras as keras

model = keras.models.Sequential([
    keras.Input(shape = (227, 227, 3)),

    #CNN
    keras.layers.Conv2D(96, (11, 11), 4, activation = 'relu'),
    keras.layers.Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=1e-4, beta=0.75, bias=2.0)),
    keras.layers.MaxPool2D((3, 3), 2),

    keras.layers.Conv2D(256, (5, 5),padding = 'same', activation='relu'),
    keras.layers.Lambda(lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=1e-4, beta=0.75, bias=2.0)),
    keras.layers.MaxPool2D((3, 3), 2),

    keras.layers.Conv2D(384, (3, 3),padding = 'same', activation='relu'),

    keras.layers.Conv2D(384, (3, 3),padding = 'same', activation='relu'),

    keras.layers.Conv2D(256, (3, 3),padding = 'same', activation='relu'),
    keras.layers.MaxPool2D((3, 3), 2),

    #ANN
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation = 'relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(4096, activation = 'relu'),
    keras.layers.Dropout(0.5),

    keras.layers.Dense(1000, activation = 'softmax')
])

model.summary()
