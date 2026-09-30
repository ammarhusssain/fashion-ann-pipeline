import os
import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

params = yaml.safe_load(open("params.yaml"))["train"]
data = np.load("data/processed/data.npz")

model = keras.Sequential([
    keras.layers.Input(shape=(28, 28)),
    keras.layers.Flatten(),                                   
    keras.layers.Dense(params["dense_units"], activation="relu"),  
    keras.layers.Dropout(params["dropout_rate"]),                  
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(data["x_train"], data["y_train"],
                    validation_data=(data["x_val"], data["y_val"]),
                    epochs=params["epochs"], batch_size=params["batch_size"])

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")
pd.DataFrame(history.history).to_csv("models/history.csv", index=False)