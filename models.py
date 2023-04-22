from joblib import dump, load
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
import evaluate
import dataset


class HeuristicClassifier:
    def __init__(self):
        pass

    def predict(self, X):
        return [self._classify(row) for idx, row in X.iterrows()]

    @staticmethod
    def _classify(row):
        elevation = float( row["Elevation"] ) 
        if elevation < 1859:
            return 0
        elif elevation < 2192:
            return 1
        elif elevation < 2525:
            return 2
        elif elevation < 2858:
            return 3
        elif elevation < 3191:
            return 4
        elif elevation < 3524:
            return 5
        else:
            return 6
    
def train_sklern_machine_lerning_model(x_train, x_test, y_train, y_test):
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)

    # Predictions on the testing set
    lr_pred = lr_model.predict(x_test)
    rf_pred = rf_model.predict(x_test)

    # Evaluate accuracy 
    lr_accuracy = accuracy_score(y_test, lr_pred)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print('Logistic Regression accuracy:', lr_accuracy)
    print('Random Forest Classifier accuracy:', rf_accuracy)

    # Save and return better model 
    model_path = "models/skl_model.joblib"
    dump(lr_model, model_path) if lr_accuracy > rf_accuracy else dump(rf_model, model_path)
    return lr_model if lr_accuracy > rf_accuracy else rf_model


def train_neural_network_model(x_train, x_test, y_train, y_test):
    hyperparameters = {'activation': ['relu', 'sigmoid', 'tanh', 'elu'],
                       'dropout': [0.0, 0.1, 0,2],
                       'layer_2nd': [64, 112, 156]}
    # Define classifier
    classifier = KerasClassifier(build_fn=create_model)

    # Testing and printing best result 
    grid = GridSearchCV(estimator=classifier, param_grid=hyperparameters, cv=5)
    grid_result = grid.fit(x_train, y_train)
    
    print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')
    best_model = grid_result.best_estimator_.model


    # Train the best model with early stopping callback
    history = best_model.fit(x_train, y_train,
                            epochs=64, batch_size=32,
                            validation_data=(x_test, y_test),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
    
    y_pred = best_model.predict( x_test )
    y_pred = np.argmax(y_pred, axis=1) 
    y_test = np.argmax(y_test, axis=1)

    # creating plots
    evaluate.create_model_plots(history)
    evaluate.create_confusion_mtx( y_pred, y_test )
    
    model_path = "models/nn_model.h5"
    best_model.save(model_path)
    return best_model

def create_model(activation='relu', dropout=0.0, layer_2nd=64):
    print('Testing hyperparameters:')
    print('activation: ', activation)
    print('dropout: ', dropout)
    print('layer_2nd: ', layer_2nd)

    model = Sequential()
    model.add(Dense(54, input_dim=16, activation=activation))
    model.add(Dense(layer_2nd, activation=activation))
    model.add(Dropout(dropout))
    model.add(Dense(7, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer='adam')
    return model

def load_sklern_ml_model(path = "models/skl_model.joblib"):
   model = load(path)
   return model
  
def load_nn_model(path = "models/nn_model.h5"):
   model = load_model(path)
   return model
  
def decode_answer(answer):
   if( answer == 0 ):
      return "Spruce/Fir"
   if( answer == 1 ):
      return "Lodgepole Pine"
   if( answer == 2 ):
      return "Ponderosa Pine"
   if( answer == 3 ):
      return "Cottonwood/Willow"
   if( answer == 4 ):
      return "Aspen"
   if( answer == 5 ):
      return "Douglas-fir"
   if( answer == 6 ):
      return "Krummholz"

def compare_models(model1, model2, model3, x_test, y_test, x_train):
    model1_pred = model1.predict(x_test)
    model2_pred = model2.predict(x_test)

    std_scaler = dataset.load_pca('models/std_scaler.joblib')
    pca = dataset.load_pca('models/pca.joblib')
    std_scaler.fit(x_train)
    pca.fit(x_train)

    x_train = std_scaler.transform(x_train)
    x_test = std_scaler.transform(x_test)
    x_train = pca.transform(x_train)
    x_test = pca.transform(x_test)
    
    model3_pred = np.argmax( model3.predict(x_test), axis=-1)
    
    model1_acc = accuracy_score(y_test, model1_pred)
    model2_acc = accuracy_score(y_test, model2_pred)
    model3_acc = accuracy_score(y_test, model3_pred)
    
    model1_f1 = f1_score(y_test, model1_pred, average='weighted')
    model2_f1 = f1_score(y_test, model2_pred, average='weighted')
    model3_f1 = f1_score(y_test, model3_pred, average='weighted')
    
    data = [['Heuristic classifier', model1_acc, model1_f1],
            ['Sk-learn model', model2_acc, model2_f1],
            ['Neural Network', model3_acc, model3_f1]]
    
    df = pd.DataFrame(data, columns=['Model', 'Accuracy', 'F1 Score'])
    return df