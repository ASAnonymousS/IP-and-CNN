import numpy
import tensorflow.keras as keras
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
from matplotlib import pyplot

x_train = numpy.loadtxt('input.csv', delimiter=',')
y_train = numpy.loadtxt('labels.csv', delimiter=',')

x_train = x_train.reshape(len(x_train),100,100,3)
y_train = y_train.reshape(len(y_train),1)

x_train = x_train / 255.0

model = models.Sequential([
    # Input Specification:
    keras.Input(shape = (100, 100, 3)),

    # Layers Addition:
    layers.Conv2D(32, (3, 3), activation = 'relu'),
    layers.MaxPool2D((2,2)),

    layers.Conv2D(32,(3,3),activation = 'relu'),
    layers.MaxPool2D((2,2)),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(x_train, y_train, epochs = 10, batch_size=64)

model.save('model.keras')

pyplot.plot(history.history['accuracy'],color='red')
pyplot.plot(history.history['loss'])
pyplot.show()
