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


def create_loss_plot(train_loss, val_loss, logs_path, epoch, index):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, linestyle='solid' ,label="Training loss")
    plt.plot(val_loss, linestyle='solid', label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.autoscale(enable=True, axis="Both")
    plt.title("Training vs. Validation loss")
    plt.legend()
    plt.savefig(os.path.join(logs_path, f"loss_plot_epoch_{epoch}_snapshot_{index}"))
    plt.show()
