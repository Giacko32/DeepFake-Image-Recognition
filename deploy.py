import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sys import argv
from prepare_dataset import img_preprocess
from cv2 import CascadeClassifier
from cv2.data import haarcascades
import os

# Carica il modello e l'eventuale scaler, oltre a specificare la modalità per lbp
with open("best_model.pkl", "rb") as f:
    data = pickle.load(f)
    if len(data) == 3:
        model, scaler, is_default = data
    else:
        model, is_default = data
        scaler = None


# Funzione per estrarre l'lbp e predire con il modello
def extract_features_and_predict(image, is_default, scaler):
    
    #calcola lbp e istogramma in base alla modalità
    if is_default:
        lbp = local_binary_pattern(image, P=256, R=1.0, method="default")
        hist, _ = np.histogram(lbp, bins=256, density=True)
    else:
        lbp = local_binary_pattern(image, P=8, R=1.0, method="uniform")
        hist, _ = np.histogram(lbp, bins=10, density=True)
    
    #fa il reshape poiché la predict si aspetta un array 2D
    hist = hist.reshape(1, -1)

    #esegue la scalatura se prevista per il best model
    if scaler is not None:
        hist = scaler.transform(hist)

    #inferenza
    prediction = model.predict(hist)

    return lbp, hist[0], prediction[0]


def main():

    #controllo sui parametri
    if len(argv) not in [2, 3]:
        print("Uso corretto: python deploy.py <path_immagine> <crop opzionale (0 o 1)>")
        return None
    
    path = argv[1]

    #parametro opzionale di crop
    crop = 1 if len(argv) == 2 else int(argv[2])

    img = None

    #controllo se il percorso esiste
    if not os.path.exists(path):
        print("Errore, percorso non trovato")
        return None

    #preprocessing dell'immagine con crop se necessario
    if crop == 1:
        face_detector = CascadeClassifier(haarcascades + "haarcascade_frontalface_alt.xml")
        img = img_preprocess(True, face_detector, path)
    else:
        img = img_preprocess(False, None, path)
    
    #nel caso di problemi con il Cascade
    if img is None:
        print("Impossibile croppare, volto non rilevato")
        return None 
    
    #estrae le features
    lbp_img, hist, pred = extract_features_and_predict(img, is_default, scaler)
    
    print(f"Predizione: {'deepfake' if pred == 1 else 'real'}")
    
    #mostra immagine, LBP e istogramma
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Immagine Preprocessata")
    axes[0].axis("off")

    axes[1].imshow(np.uint8(lbp_img), cmap="gray")
    axes[1].set_title("Immagine LBP")
    axes[1].axis("off")
    
    axes[2].bar(np.arange(len(hist)), hist, width=0.8, color='#CF2A2A', edgecolor='black')
    axes[2].set_title("Istogramma LBP")
    axes[2].set_xlabel("Valori LBP")
    axes[2].set_ylabel("Value")

    plt.suptitle(f"Predizione: {'deepfake' if pred == 1 else 'real'}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()