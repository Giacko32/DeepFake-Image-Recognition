from featureExtractionTools import FeatureExtraction
from sys import argv
import os

def main():
    
    #verifica i parametri in ingresso
    if len(argv) != 2:
        print("Uso: python extract_features.py <percorso_dataset_preprocessato>")
        return None
    
    #percorso della cartella in cui si trova l'intero dataset
    path = argv[1]

    #controllo se il percorso esiste
    if not os.path.exists(path):
        print("Errore, percorso non trovato")
        return None

    #splitting dei soggetti in train, validation e test
    train, validation, test = FeatureExtraction.load_split_dataset(path)

    #estrazione delle features dalle immagini con lo stesso splitting, nelle due modalità
    train_set_uniform, validation_set_uniform, test_set_uniform = FeatureExtraction.extract_features(path, "uniform", train, validation, test)
    train_set_default, validation_set_default, test_set_default = FeatureExtraction.extract_features(path, "default", train, validation, test)
    
    #crea la cartella in cui salvare le features, se non esiste
    os.makedirs("features", exist_ok=True)

    #salva le features con pickle
    FeatureExtraction.save_features("features", "uniform", train_set_uniform, validation_set_uniform, test_set_uniform)
    FeatureExtraction.save_features("features", "default", train_set_default, validation_set_default, test_set_default)


if __name__ == "__main__":
    main()