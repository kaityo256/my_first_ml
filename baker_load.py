import numpy as np
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
        if np.random.random() < 0.5:
            x.append(get_random(n))
            y.append(0)
        else:
            x.append(get_baker(n))
            y.append(1)
    x = np.array(x)
    y = np.array(y)
    return x, y


model = keras.models.load_model("baker.keras")

all_random_data = np.array([get_random(100) for _ in range(100)])
all_random_labels = np.array([0] * 100)

r_loss, r_acc = model.evaluate(all_random_data, all_random_labels)

print("When everything is random")
print(f"Test Loss = {r_loss}")
print(f"Test Accuracy = {r_acc}")

all_baker_data = np.array([get_baker(100) for _ in range(100)])
all_baker_labels = np.array([1] * 100)

b_loss, b_acc = model.evaluate(all_baker_data, all_baker_labels)

print("When everything is baker map")
print(f"Test Loss = {b_loss}")
print(f"Test Accuracy = {b_acc}")
