import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_data():
    train_data, test_data = keras.datasets.mnist.load_data()
    train_images, train_labels = train_data
    test_images, test_labels = test_data
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return (train_images, train_labels, test_images, test_labels)


(train_images, train_labels, test_images, test_labels) = get_data()

model = keras.models.load_model("model.keras")

predictions = model.predict(test_images[0:20])

for i in range(20):
    predicted_index = np.argmax(predictions[i])
    print(f"prediction= {predicted_index} answer = {test_labels[i]}")
