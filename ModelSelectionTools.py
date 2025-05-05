from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import utils
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class ModelSelectionTools:

    #metodo per preprocessing dei dati
    @staticmethod
    def preprocessData(train_set, validation_set, test_set):
        #scaling su tutti i set ma a partire dalla media e std del train set
        std_scaler = StandardScaler()
        std_scaler.fit(train_set[0])

        scaled_train_set = (std_scaler.transform(train_set[0]), train_set[1])
        scaled_validation_set = (std_scaler.transform(validation_set[0]), validation_set[1])
        scaled_test_set = (std_scaler.transform(test_set[0]), test_set[1])
        
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
    def evaluateOnSet(trained_model, set):
        true_labels = set[1]
        predicted_labels = trained_model.predict(set[0])
        accuracy = accuracy_score(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)
        return accuracy, cm
    