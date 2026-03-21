import tensorflow.keras as keras
from matplotlib import pyplot

(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = keras.models.Sequential([
    keras.Input(shape = (32, 32, 3)),

    #CNN
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),

    #ANN
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

model.summary()

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, Y_train, epochs = 10)

print(history)

model.save('model.keras')

pyplot.plot(history.history['accuracy'],color='red')
pyplot.plot(history.history['loss'])
pyplot.show()
