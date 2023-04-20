import dataset
import models

if __name__ == "__main__":
    # Download, preprocess and read dataset
    data = dataset.load_data()
    x_train, x_test, y_train, y_test = dataset.preprocess_baseline_data(data)
    x_ohe, y_ohe  = dataset.preprocess_one_hot_data(data)
    
    # Train, evaluate and save models
    models.train_sklern_machine_lerning_model(x_train, x_test, y_train, y_test)
    models.train_neural_network_model(x_ohe, y_ohe)
