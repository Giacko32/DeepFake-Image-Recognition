from feature_extractionTools import FeatureExtraction
import pickle
from sys import argv

def main():
    
    #verifica i parametri in ingresso
    if len(argv) != 3:
        print("Parametri errati")
        return None
    
    #percorso della cartella in cui si trova l'intero dataset
    path = argv[1]

    #splitting dei soggetti in train, validation e test
    train, validation, test = FeatureExtraction.load_split_dataset(path)

    #modalità di esecuzione del local binary pattern
    mode = argv[2]

    #estrazione delle feature per il train set
    train_set = FeatureExtraction.extract_features_from_set(path, train, mode)
    print("train set concluso: ", train_set[0].shape)

    #estrazione delle feature per il validation set
    validation_set = FeatureExtraction.extract_features_from_set(path, validation, mode)
    print("validation set concluso: ", validation_set[0].shape)

    #estrazione delle feature per il test set
    test_set = FeatureExtraction.extract_features_from_set(path, test, mode)
    print("test set concluso: ", test_set[0].shape)

    #salvataggio dei dati di training
    with open('train.pkl', 'wb') as f:
        pickle.dump(train_set, f)

    #salvataggio dei dati di validazione
    with open('val.pkl', 'wb') as f:
        pickle.dump(validation_set, f)

    #salvataggio dei dati di test
    with open('test.pkl', 'wb') as f:
        pickle.dump(test_set, f)


main()