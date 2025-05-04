import pickle
import numpy as np
from ModelSelectionTools import ModelSelectionTools

def main():

    #caricamento di train, validation e test set con modalità uniform
    with open("features/train_uniform.pkl", "rb") as f:
        train_set = pickle.load(f)
        
    with open("features/val_uniform.pkl", "rb") as f:
        validation_set = pickle.load(f)

    with open("features/test_uniform.pkl", "rb") as f:
        test_set = pickle.load(f)

    #caricamento di train, validation e test set con modalità default
    # with open("train_default.pkl", "rb") as f:
    #     train_set = pickle.load(f)
        
    # with open("val_default.pkl", "rb") as f:
    #     validation_set = pickle.load(f)

    # with open("test_default.pkl", "rb") as f:
    #     test_set = pickle.load(f)

    #pre-processing
    train_scaled, validation_scaled, test_scaled = ModelSelectionTools.preprocessData(train_set, validation_set, test_set, scaling=True, apply_pca=False)
    train, validation, test = ModelSelectionTools.preprocessData(train_set, validation_set, test_set, scaling=False, apply_pca=False)

    #model selection in base alla metrica di accuracy
    best_model = None
    best_accuracy = None
    best_is_scaled = False

    for model in ["RandomForest", "LogRegression", "SVC"]:
        for scale in [True, False]:
            trained_model = None
            if scale:
                trained_model = ModelSelectionTools.train_model(model, train_scaled)
                acc, _ = ModelSelectionTools.evaluateOnSet(trained_model, validation_scaled)
            else:
                trained_model = ModelSelectionTools.train_model(model, train)
                acc, _ = ModelSelectionTools.evaluateOnSet(trained_model, validation)

            print(f"{model} {'con scaling' if scale else 'senza scaling'}: {acc*100}%")
            
            if best_model == None:
                best_model = trained_model
                best_accuracy = acc
                best_is_scaled = scale
            elif acc > best_accuracy:
                best_model = trained_model
                best_accuracy = acc
                best_is_scaled = scale


    print(f"Il miglior modello è stato : {best_model} {'con scaling' if best_is_scaled else 'senza scaling'} con una accuracy del {best_accuracy*100:3.4f}%")

    if best_is_scaled:
        test_acc, conf_matrix = ModelSelectionTools.evaluateOnSet(best_model, test_scaled)
    else:
        test_acc, conf_matrix = ModelSelectionTools.evaluateOnSet(best_model, test)

    print(f"Il miglior modello ha ottenuto una accuracy del {test_acc*100:3.4f}% sul test set, matrice di confusione:")
    print(conf_matrix)

    with open("best_model.pkl", "wb") as f:
        pickle.dump(best_model,f)
main()