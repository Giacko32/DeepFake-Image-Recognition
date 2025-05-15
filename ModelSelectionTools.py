from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle, os


class ModelSelectionTools:

    #metodo per caricare train, validation e test set di una specifica modalità
    @staticmethod
    def load_dataset(mode):
        with open(os.path.join("features",f"train_{mode}.pkl"), "rb") as f:
            train = pickle.load(f)

        with open(os.path.join("features",f"val_{mode}.pkl"), "rb") as f:
            val = pickle.load(f)
            
        with open(os.path.join("features",f"test_{mode}.pkl"), "rb") as f:
            test = pickle.load(f)
        return train, val, test
    
    #metodo per preprocessing dei dati(scaling)
    @staticmethod
    def preprocessData(train_set, validation_set, test_set):
        #scaling su tutti i set ma a partire dalla media e std del train set
        std_scaler = StandardScaler()
        std_scaler.fit(train_set[0])

        scaled_train_set = (std_scaler.transform(train_set[0]), train_set[1])
        scaled_validation_set = (std_scaler.transform(validation_set[0]), validation_set[1])
        scaled_test_set = (std_scaler.transform(test_set[0]), test_set[1])
        
        #restituisce lo scaler per l'utilizzo nel deploy
        return scaled_train_set, scaled_validation_set, scaled_test_set, std_scaler

    
    #addestramento di un modello selezionato
    @staticmethod
    def train_model(model_chosen, train_set):
        
        if model_chosen == "RandomForest":
            randomForest = RandomForestClassifier()
            randomForest.fit(train_set[0], train_set[1])
            return randomForest
        
        elif model_chosen == "LogRegression":
            logisticRegression = LogisticRegression()
            logisticRegression.fit(train_set[0], train_set[1])
            return logisticRegression
        
        elif model_chosen == "SVC":
            svc = LinearSVC()
            svc.fit(train_set[0], train_set[1])
            return svc
        
    #metodo di valutazione di un modello addestrato
    @staticmethod
    def evaluateOnSet(trained_model, data_set):
        true_labels = data_set[1]
        predicted_labels = trained_model.predict(data_set[0])
        #utilizzo la balanced_accuracy per compensare lo sbilanciamento delle classi
        accuracy = balanced_accuracy_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        return accuracy, cm
    