import os, random
from skimage import io, feature
import numpy as np

class FeatureExtraction():

    #metodo per caricare e splittare i soggetti 
    @staticmethod
    def load_split_dataset(path):
        #carica la lista di soggetti (cartelle) dal dataset
        subjects = os.listdir(path)

        #shuffle per mescolare soggetti fake e real
        random.seed(42)
        random.shuffle(subjects)

        #seleziona il 60% di soggetti per il training
        train_subjects = subjects[0:int(0.6*len(subjects))]

        #seleziona il 20% di soggetti per il validation
        validation_subjects = subjects[int(0.6*len(subjects)):int(0.8*len(subjects))]

        #seleziona il restante 20% di soggetti per il test
        test_subjects = subjects[int(0.8*len(subjects)):]

        return train_subjects, validation_subjects, test_subjects
    
    #metodo per estrarre il vettore di (features, label) da un set
    @staticmethod
    def extract_features_from_set(path, subjects_set, mode):
        
        features = []
        labels = []

        #per ogni soggetto (cartella) del set 
        for subject in subjects_set:

            read_path = f"{path}/{subject}"

            #stabilisce in base al nome della cartella la label
            label = 1 if subject.split("_")[0] == "fake" else 0
            
            #per ogni immagine del soggetto
            for img in os.listdir(read_path):

                #carica l'immagine in scala di grigi
                img = io.imread(read_path + "/" + img, as_gray=True)

                #esegue l'estrazione delle features in base alla modalità
                lbp = None
                if mode == "uniform":
                    lbp = feature.local_binary_pattern(img, P=8, R=1.0, method="default")
                elif mode == "default":
                    lbp = feature.local_binary_pattern(img, P=256, R=1.0, method="default")

                #crea l'istogramma
                hist, _ = np.histogram(lbp, bins=256, density=True)
                
                #salva sia il vettore di features generato, sia il la label, in modo da mantenere la corrispondenza
                features.append(hist)
                labels.append(label)
        
        #converte features e labels in array numpy
        features = np.array(features)
        labels = np.array(labels)

        #ritorna la tupla (X, y)
        return (features, labels)

                
