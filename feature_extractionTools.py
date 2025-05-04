import os, random
from skimage import io, feature
import numpy as np

class FeatureExtraction():

    #metodo per caricare e splittare i soggetti 
    @staticmethod
    def load_split_dataset(path):
        # carica la lista di soggetti (cartelle) dal dataset
        subjects = os.listdir(path)

        #seleziona i soggetti real e deepfake
        real_subjects = [s for s in subjects if s.startswith("real_")]
        fake_subjects = [s for s in subjects if s.startswith("fake_")]

        #shuffle 
        random.seed(42)
        random.shuffle(real_subjects)
        random.shuffle(fake_subjects)

        #metodo per splittare i soggetti in 60% train, 20% validation e 20% test
        def split(lista):
            n = len(lista)
            train = lista[:int(0.6 * n)]
            val = lista[int(0.6 * n):int(0.8 * n)]
            test = lista[int(0.8 * n):]
            return train, val, test

        #ottengo i sottoinsiemi splittati di soggetti real e fake
        real_train, real_val, real_test = split(real_subjects)
        fake_train, fake_val, fake_test = split(fake_subjects)

        #unisco i soggetti real e fake nei train, validation e test set
        train_subjects = real_train + fake_train
        validation_subjects = real_val + fake_val
        test_subjects = real_test + fake_test

        #shuffle per mescolare ulteriormente
        random.shuffle(train_subjects)
        random.shuffle(validation_subjects)
        random.shuffle(test_subjects)

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

                
