import pickle
import numpy as np
from ModelSelectionTools import ModelSelectionTools

def main():

    #caricamento di train, validation e test set
    with open("train.pkl", "rb") as f:
        train_set = pickle.load(f)
        
    with open("val.pkl", "rb") as f:
        validation_set = pickle.load(f)

    with open("test.pkl", "rb") as f:
        test_set = pickle.load(f)

    #pre-processing
    train_set, validation_set, test_set = ModelSelectionTools.preprocessData(train_set, validation_set, test_set, scaling=True, whitening=False)


    #model selection in base alla metrica di accuracy
    best_model = None
    best_accuracy = None

    for model in ["RandomForest", "LogRegression", "SVC"]:
        trained_model = ModelSelectionTools.train_model(model, train_set)
        acc, _ = ModelSelectionTools.evaluateOnSet(trained_model, validation_set)
        print(f"{model}: {acc*100}%")
        if best_model == None:
            best_model = trained_model
            best_accuracy = acc
        elif acc > best_accuracy:
            best_model = trained_model
            best_accuracy = acc


    print(f"Il miglior modello è stato : {best_model} con una accuracy del {best_accuracy*100:3.4f}%")

    test_acc, conf_matrix = ModelSelectionTools.evaluateOnSet(best_model, test_set)

    print(f"Il miglior modello ha ottenuto una accuracy del {test_acc*100:3.4f}% sul test set, matrice di confusione:")
    print(conf_matrix)

main()