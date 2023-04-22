import matplotlib.pyplot as plt
import seaborn as sns
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
    plt.savefig("evaluate/Model_Accuracy.png")
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_title('Model Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['train', 'val'], loc='upper left')
    plt.savefig("evaluate/Model_Loss.png")
    plt.clf()

    return True

def create_confusion_mtx(y_pred, y_test):
    confusion_mtx = confusion_matrix( y_test, y_pred, normalize='true')
    fig, ax = plt.subplots()
    ax = sns.heatmap(confusion_mtx, annot=True, cmap='Blues')
    plt.savefig("evaluate/Confusion_Matrix.png")
