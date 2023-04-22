# Covertype_with_RestAPI
This project aims to classify the Covertype dataset using various machine learning models and neural networks. The project also includes a simple REST API to serve the trained models.

## Dataset
The dataset used in this project is the Covertype dataset, which can be found on the UCI Machine Learning Repository. It contains information about forest cover types based on cartographic variables.

## Implementation
A simple heuristic has been implemented to classify the data.
Two simple machine learning models have been trained using Scikit-learn lib.
A neural network has been trained using TensorFlow library. A function has been created to find the best set of hyperparameters for the neural network, and training curves have been plotted.

A REST API has been created to serve the trained models. With simple html page user can choose a model (heuristic, better Scikit-learn model, or neural network) and data to predict to get a response.
Usage
To run the project, follow these steps:

Clone the repository.
Install the necessary packages by running 
```
$ pip install -r requirements.txt'.
```
Run the Flask app by executing 
```
$ python main.py 
$ python app.py 
```
in the project directory. After that, you can open index.html insert data to predict.   