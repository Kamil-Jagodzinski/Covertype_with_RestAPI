import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def create_model_plots(history):
    # Plot training curves
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_title('Model Accuracy')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'val'], loc='upper left')
    plt.savefig("img/Model Accuracy.png")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'val'], loc='upper left')
    plt.savefig("img/Model Loss.png")
    plt.clf()

    return True

def create_confusion_mtx(y_pred, y_test):
    confusion_mtx = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
    ((confusion_mtx.astype('float') / confusion_mtx.sum(axis=1)[:, np.newaxis]) * 100).round(2)
    fig, ax = plt.subplots()
    ax = sns.heatmap(confusion_mtx, annot_kws={"size": 20})
    plt.savefig("Confusion Matrix.png")