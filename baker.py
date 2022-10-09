import numpy as np
import tensorflow as tf
from tensorflow import keras


def get_random(length):
    return np.random.random(length)


def get_baker(length):
    a = np.zeros(length)
    x = np.random.random()
    for i in range(length):
        x = x * 3.0
        x = x - int(x)
        a[i] = x
    return a


def make_data(n, length):
    x = []
    y = []
    for _ in range(length):
        if(np.random.random() < 0.5):
            x.append(get_random(n))
            y.append(0)
        else:
            x.append(get_baker(n))
            y.append(1)
    x = np.array(x)
    y = np.array(y)
    return x, y


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(100),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


n = 100
train_data, train_labels = make_data(n, 60000)
test_data, test_labels = make_data(n, 10000)

model = create_model()

model.fit(train_data, train_labels, epochs=5)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print(f"Test Loss = {test_loss}")
print(f"Test Accuracy = {test_acc}")

model.save_weights('baker')
