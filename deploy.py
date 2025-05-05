import os
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import local_binary_pattern

# Carica il modello
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)
    if len(data) == 3:
        model, scaler, is_default = data
    else:
        model, is_default = data
        scaler = None

# Funzione per elaborare un'immagine (LBP, features, predizione)
def process_image(image_path, is_default, scaler):
    image = imread(image_path)

    if is_default:
        lbp = local_binary_pattern(image, P=256, R=1.0, method="default")
    else:
        lbp = local_binary_pattern(image, P=8, R=1.0, method="uniform")

    hist, _ = np.histogram(lbp, bins=256, density=True)
    features = hist.reshape(1, -1)

    if scaler is not None:
        features = scaler.transform(features)

    prediction = model.predict(features)
    return image, hist, prediction[0]

# Seleziona un'immagine fake
root_dir = "processed_dataset"
fake_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and "fake" in d.lower()]
real_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and "real" in d.lower()]

fake_folder = random.choice(fake_dirs)
real_folder = random.choice(real_dirs)

fake_image_path = os.path.join(fake_folder, random.choice([f for f in os.listdir(fake_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))
real_image_path = os.path.join(real_folder, random.choice([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]))

# Processa entrambe
gray_fake, hist_fake, pred_fake = process_image(fake_image_path, is_default, scaler)
gray_real, hist_real, pred_real = process_image(real_image_path, is_default, scaler)

# Visualizzazione
plt.figure(figsize=(10, 8))

# Immagine fake
plt.subplot(2, 2, 1)
plt.imshow(gray_fake, cmap="gray")
plt.title(f"FAKE - Predizione: {'deepfake' if pred_fake == 1 else 'real'}")
plt.axis("off")

# Istogramma fake
plt.subplot(2, 2, 2)
plt.plot(hist_fake)
plt.title("Istogramma LBP - Fake")
plt.xlabel("Valore LBP")
plt.ylabel("Densità")

# Immagine real
plt.subplot(2, 2, 3)
plt.imshow(gray_real, cmap="gray")
plt.title(f"REAL - Predizione: {'deepfake' if pred_real == 1 else 'real'}")
plt.axis("off")

# Istogramma real
plt.subplot(2, 2, 4)
plt.plot(hist_real)
plt.title("Istogramma LBP - Real")
plt.xlabel("Valore LBP")
plt.ylabel("Densità")

plt.tight_layout()
plt.show()
