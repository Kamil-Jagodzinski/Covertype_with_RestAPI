from joblib import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import images

def heuristic_classifier(row):
    elevation = row['Elevation']
    if elevation < 1859:
      return 1
    elif elevation < 2192:
      return 2
    elif elevation < 2525:
      return 3
    elif elevation < 2858:
      return 4
    elif elevation < 3191:
      return 5
    elif elevation < 3524:
      return 6
    else:
      return 7
    
def train_sklern_machine_lerning_model(x_train, x_test, y_train, y_test):
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)
    rf_model = RandomForestClassifier(n_estimators=100)
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


def train_neural_network_model(x_data, y_data):
    hyperparameters = {'optimizer': ['adam', 'sgd', 'rmsprop'],
                       'activation': ['relu', 'tanh', 'sigmoid'],
                       'layer_1st': [86, 112, 138],
                       'layer_2nd': [32, 44, 56]}
    # Define classifier
    classifier = KerasClassifier(build_fn=create_model)

    # Testing and printing best result 
    grid = GridSearchCV(estimator=classifier, param_grid=hyperparameters, cv=5)
    grid_result = grid.fit(x_data, y_data)
    
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    best_model = grid_result.best_estimator_.model

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=25,stratify=y_data)

    # Train the best model with early stopping callback
    history = best_model.fit(x_train, y_train,
                            epochs=50, batch_size=32,
                            validation_data=(x_test, y_test),
                            callbacks=[EarlyStopping(monitor='val_loss', patience=10)])
    
    y_pred = best_model.predict(x_test)

    # creating plots
    images.create_model_plots(history)
    images.create_confusion_mtx(y_pred, y_test)
    
    model_path = "models/nn_model.h5"
    best_model.save(model_path)
    return best_model

def create_model(optimizer='adam', activation='relu', layer_1st=86, layer_2nd=32):
    print('Testing hyperparameters:')
    print('optimizer: ', optimizer)
    print('activation: ', activation)
    print('layer_1st: ', layer_1st)
    print('layer_2nd: ', layer_2nd)

    model = Sequential()
    model.add(Dense(layer_1st, input_dim=54, activation=activation))
    model.add(Dense(layer_2nd, activation=activation))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                  optimizer=optimizer)
    return model


def load_sklern_ml_model(path = "models/skl_model.joblib"):
   return 0
  
def load_nn_model(path = "models/nn_model.h5"):
   return 0
  
