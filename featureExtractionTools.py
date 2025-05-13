import os, random
from skimage import io, feature
import numpy as np
import pickle

class FeatureExtraction():

    #metodo per caricare e splittare i soggetti 
    @staticmethod
    def load_split_dataset(path):
        # carica la lista di soggetti (cartelle) dal dataset
        subjects = os.listdir(path)

        #seleziona i soggetti real e deepfake
        real_subjects = [s for s in subjects if s.startswith("real_")]
        fake_subjects = [s for s in subjects if s.startswith("fake_")]

        #shuffle ma deterministico per il seed
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

        #unisco i soggetti real e fake nei train, validation e test set finali
        train_subjects = real_train + fake_train
        validation_subjects = real_val + fake_val
        test_subjects = real_test + fake_test

        #shuffle per mescolare i real e i fake
        random.shuffle(train_subjects)
        random.shuffle(validation_subjects)
        random.shuffle(test_subjects)

        return train_subjects, validation_subjects, test_subjects
        
    #metodo per estrarre il vettore (features, label) da un set
    @staticmethod
    def extract_features_from_set(path, subjects_set, mode):
        
        features = []
        labels = []

        #per ogni soggetto (cartella) del set 
        for subject in subjects_set:

            read_path = os.path.join(path, subject)

            #stabilisce in base al nome della cartella la label
            label = 1 if subject.split("_")[0] == "fake" else 0
            
            #per ogni immagine del soggetto
            for img_name in os.listdir(read_path):

                #carica l'immagine in scala di grigi
                img = io.imread(os.path.join(read_path, img_name), as_gray=True)

                #esegue l'estrazione delle features in base alla modalità
                lbp = None
                if mode == "uniform":
                    lbp = feature.local_binary_pattern(img, P=8, R=1.0, method="uniform")
                elif mode == "default":
                    lbp = feature.local_binary_pattern(img, P=256, R=1.0, method="default")
                
                n_bins = 10 if mode == "uniform" else 256

                #crea l'istogramma, usando density = True per evitare che le immagini
                #più grandi generino istogrammi con conteggi più alti
                hist, _ = np.histogram(lbp, bins=n_bins, density=True)

                #salva sia il vettore di features generato, sia il la label, in modo da mantenere la corrispondenza
                features.append(hist)
                labels.append(label)
        
        #converte features e labels in array numpy
        features = np.array(features)
        labels = np.array(labels)

        #ritorna la tupla (X, y)
        return (features, labels)
    
    #metodo per automatizzare l'estrazione delle features in una modalità specifica
    @staticmethod
    def extract_features(path, mode, train, validation, test):
        #estrazione delle feature per il train set
        train_set = FeatureExtraction.extract_features_from_set(path, train, mode)
        print("train set concluso: ", train_set[0].shape)

        #estrazione delle feature per il validation set
        validation_set = FeatureExtraction.extract_features_from_set(path, validation, mode)
        print("validation set concluso: ", validation_set[0].shape)

        #estrazione delle feature per il test set
        test_set = FeatureExtraction.extract_features_from_set(path, test, mode)
        print("test set concluso: ", test_set[0].shape)

        return train_set, validation_set, test_set
    
    #metodo per automatizzare il salvataggio delle features in una modalità specifica
    @staticmethod
    def save_features(path, lbp_mode, train, validation, test):
        #salvataggio dei dati di training
        with open(os.path.join(path, f"train_{lbp_mode}.pkl"), 'wb') as f:
            pickle.dump(train, f)

        #salvataggio dei dati di validazione
        with open(os.path.join(path, f"val_{lbp_mode}.pkl"), 'wb') as f:
            pickle.dump(validation, f)

        #salvataggio dei dati di test
        with open(os.path.join(path, f"test_{lbp_mode}.pkl"), 'wb') as f:
            pickle.dump(test, f)

                
