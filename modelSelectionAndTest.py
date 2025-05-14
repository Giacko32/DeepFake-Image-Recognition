import pickle
from ModelSelectionTools import ModelSelectionTools
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():

    #caricamento di train, validation e test set con modalità uniform e default
    train_set_uniform, validation_set_uniform, test_set_uniform = ModelSelectionTools.load_dataset("uniform")
    train_set_default, validation_set_default, test_set_default = ModelSelectionTools.load_dataset("default")

    #pre-processing
    train_uniform_scaled, validation_uniform_scaled, test_uniform_scaled, scaler_uniform = ModelSelectionTools.preprocessData(train_set_uniform, validation_set_uniform, test_set_uniform)
    train_default_scaled, validation_default_scaled, test_default_scaled, scaler_default = ModelSelectionTools.preprocessData(train_set_default, validation_set_default, test_set_default)

    #model selection in base alla metrica di accuracy
    best_model = None
    best_accuracy = -1.0
    best_is_scaled = False
    best_is_default = False
    best_scaler = None

    for model in ["RandomForest", "LogRegression", "SVC"]:  
        for mode in ["uniform", "default"]:
            for scale in [True, False]:

                trained_model = None
                cm = None
                if scale:
                    if mode == "uniform":
                        trained_model = ModelSelectionTools.train_model(model, train_uniform_scaled)
                        acc, cm = ModelSelectionTools.evaluateOnSet(trained_model, validation_uniform_scaled)
                    else:
                        trained_model = ModelSelectionTools.train_model(model, train_default_scaled)
                        acc, cm = ModelSelectionTools.evaluateOnSet(trained_model, validation_default_scaled)
                else:
                    if mode == "uniform":
                        trained_model = ModelSelectionTools.train_model(model, train_set_uniform)
                        acc, cm = ModelSelectionTools.evaluateOnSet(trained_model, validation_set_uniform)
                    else:
                        trained_model = ModelSelectionTools.train_model(model, train_set_default)
                        acc, cm = ModelSelectionTools.evaluateOnSet(trained_model, validation_set_default)

                print(f"Il modello {model}, usando {mode}, con scaling {scale} ha ottenuto {acc*100:4.4f}")
                
                if acc > best_accuracy:
                    best_model = trained_model
                    best_accuracy = acc
                    best_is_scaled = scale
                    best_is_default = True if mode == "default" else False
                    if best_is_scaled:
                        best_scaler = scaler_default if best_is_default else scaler_uniform


    print(f"Il miglior modello è stato : {best_model} {'con scaling' if best_is_scaled else 'senza scaling'} {'con modalità default' if best_is_default else 'con modalità uniform' } con una accuracy del {best_accuracy*100:3.4f}%")

  #testing
  #seleziona il test set uniform o default
    if best_is_default:
        test = test_default_scaled if best_is_scaled else test_set_default
    else:
        test = test_uniform_scaled if best_is_scaled else test_set_uniform

    #valuta sul test set
    test_acc, conf_matrix = ModelSelectionTools.evaluateOnSet(best_model, test)
    print(f"\nIl miglior modello ha ottenuto una accuracy del {test_acc*100:3.4f}% sul test set.")
    print("Matrice di confusione:")
    
    ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Real", "Deepfake"]).plot(cmap="Reds")
    plt.title("Matrice di Confusione - Test Set")
    plt.show()

    # Salvataggio del miglior modello insieme allo scaler e alla modalità
    with open("best_model.pkl", "wb") as f:
        if best_is_scaled:
            to_save = (best_model, scaler_default if best_is_default else scaler_uniform, best_is_default)
            pickle.dump(to_save, f)
        else:
            pickle.dump((best_model, best_is_default), f)
        

if __name__ == "__main__":
    main()