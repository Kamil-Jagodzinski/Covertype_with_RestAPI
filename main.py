import dataset
import models

if __name__ == "__main__":
    # Download, preprocess and read dataset
    data = dataset.load_data()

    # Train, evaluate and save models
    x_train_ohe, x_test_ohe, y_train_ohe, y_test_ohe = dataset.preprocess_one_hot_data(data)
    # nn = models.train_neural_network_model(x_train_ohe, x_test_ohe, y_train_ohe, y_test_ohe)
    x_train, x_test, y_train, y_test = dataset.preprocess_data(data)    
    # sk = models.train_sklern_machine_lerning_model(x_train, x_test, y_train, y_test)
    heuristic = models.HeuristicClassifier()

    sk = models.load_sklern_ml_model()
    nn = models.load_nn_model()

    df = models.compare_models(heuristic, sk, nn, x_test, y_test, x_train)
    df.to_csv('evaluate/comparision_results.csv', index=False)

