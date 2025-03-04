import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def create_confusion_matrix(ground_truth, predictions, log_path, nth_split):
    # Create the confusion matrix
    confusion = confusion_matrix(ground_truth, predictions.argmax(1))
    # accuracy = np.mean(confusion.diagonal().astype(float) / (confusion.sum(axis=1)))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
    disp.plot()
    disp.figure_.savefig(os.path.join(log_path, f'Confusion_matrix_split_{nth_split}.png'))
    plt.show()


def create_loss_plot(train_loss, validation_loss, log_path, nth_split):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.autoscale(enable=True, axis='Both')
    plt.legend()
    plt.savefig(os.path.join(log_path, f'loss_plot_split_{nth_split}.png'))
    plt.show()
