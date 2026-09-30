import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

matplotlib.use("Agg")

data = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")

loss, acc = model.evaluate(data["x_test"], data["y_test"], verbose=0)
y_pred = model.predict(data["x_test"]).argmax(axis=1)

classes = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]

cm = confusion_matrix(data["y_test"], y_pred)
ConfusionMatrixDisplay(cm, display_labels=classes).plot(xticks_rotation=45)
plt.tight_layout()
plt.savefig("confusion_matrix.png")

json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"), indent=2)

print(f"Test accuracy: {acc:.4f}")