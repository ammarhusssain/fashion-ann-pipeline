import tensorflow as tf
import os
import numpy as np

print("Preparing the Fashion MNIST dataset...")
os.makedirs("data/raw", exist_ok=True)


print("Loading Fashion MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
print("Dataset loaded successfully.")

np.savez("data/raw/fashion_mnist.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

print("Saved raw data:", x_train.shape, x_test.shape)