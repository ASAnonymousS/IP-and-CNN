import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot

model = keras.models.load_model("./model_Batch_Normalization_and_Dropout.keras")

test_dataset = keras.utils.image_dataset_from_directory(
    directory = './tes',
    labels = 'inferred',
    label_mode = 'int',
    batch_size = 32,
    image_size = (256,256)
)


def process(image,label):
    image = tf.cast(image / 255., tf.float32)
    return image,label


test_dataset = test_dataset.map(process)
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
