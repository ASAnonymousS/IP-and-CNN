from tensorflow import keras
import numpy

model = keras.models.load_model("./model.keras")

x_test = numpy.loadtxt('input_test.csv', delimiter=',')
y_test = numpy.loadtxt('labels_test.csv', delimiter=',')

x_test = x_test.reshape(len(x_test),100,100,3)
y_test = y_test.reshape(len(y_test),1)

x_test = x_test / 255.0

loss, accuracy = model.evaluate(x_test,y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
