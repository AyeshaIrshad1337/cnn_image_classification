import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(history):
    """
    Plot the accuracy of the model.
    Args:
        history (tf.keras.History): The history of the training.
    Returns:
        No
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
def plot_loss(history):
    """
    Plot the loss of the model.
    Args:
        history (tf.keras.History): The history of the training.
    Returns:
        No
    """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()