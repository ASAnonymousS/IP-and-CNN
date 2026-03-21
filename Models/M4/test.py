from tensorflow import keras

model = keras.models.load_model("./model.keras")

(X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()

X_test = X_test / 255.0

loss, accuracy = model.evaluate(X_test,Y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
