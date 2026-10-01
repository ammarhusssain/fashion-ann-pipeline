import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open("params.yaml"))["preprocess"]

raw = np.load("data/raw/fashion_mnist.npz")

mean = raw["x_train"].mean()
std  = raw["x_train"].std()

x_train = raw["x_train"] / std
x_test = raw["x_test"] / std

x_train, x_val, y_train, y_val = train_test_split(
    x_train, raw["y_train"],
    test_size=params["test_size"],
    random_state=params["seed"],
    stratify=raw["y_train"]
    )

os.makedirs("data/processed", exist_ok=True)

np.savez("data/processed/data.npz",
         x_train=x_train, y_train=y_train,
         x_val=x_val, y_val=y_val,
         x_test=x_test, y_test=raw["y_test"])
         
print("Train/Val/Test:", len(x_train), len(x_val), len(x_test))