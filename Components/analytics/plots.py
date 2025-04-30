import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def create_confusion_matrix(ground_truth, predictions, log_path, nth_split):
    # Create the confusion matrix
    confusion = confusion_matrix(ground_truth, predictions.argmax(1))
    # accuracy = np.mean(confusion.diagonal().astype(float) / (confusion.sum(axis=1)))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
    disp.plot()
    disp.figure_.savefig(os.path.join(log_path, f'Confusion_matrix_split_{nth_split}.png'))
    plt.show()


def create_loss_plot(args, train_loss, val_loss, logs_path, fold_id, epoch, name, save=False, show=False):
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss[1:], linestyle='solid', label="Training loss")
    plt.plot(val_loss[1:], linestyle='solid', label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.autoscale(enable=True, axis="Both")
    plt.title(f"{name}, Training vs. Validation loss, lr: {args.lr}, optimizer: {args.optimizer}")
    plt.legend()
    if save:
        plt.savefig(os.path.join(logs_path, "Figures", f"{name}_loss_foldid{fold_id}_epoch{epoch}"))
    if show:
        plt.show(block=False)
        plt.pause(8)
        plt.close()


def create_acc_plot(args, logs_path, name, foldid, epoch, acc_list, f1_list, save=False, show=False):

    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(acc_list[1:])
    plt.plot(f1_list[1:])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.autoscale(enable=True, axis='Both')
    plt.legend()
    plt.title(f'{name}, F1-Score and accuracy on validation dataset, lr: {args.lr}, optimizer: {args.optimizer}')
    if save:
        plt.savefig(os.path.join(logs_path, "Figures", f"{name}_accuracy_fold{foldid}_epoch{epoch}"))
    if show:
        plt.show(block=False)
        plt.pause(5)
        plt.close(0)

def combined_plot(logs_path, spsht_index):

    plot_data = []
    for file in os.listdir(logs_path):
        if file.split('_')[0] == "Results":
            fold_data = pd.read_csv(os.path.join(logs_path, file))
            fold_loss = fold_data['Training loss']
            plot_data.append(fold_loss)


    plt.style.use('dark_background')
    plt.figure(figsize=(10, 5))
    plt.plot(plot_data)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.autoscale(enable=True, axis="Both")

    plt.title('Combined Training vs. Validation loss')
    plt.legend()

    plt.savefig(os.path.join(logs_path, "Figures", f"folds_losses_snapshot{spsht_index}"))
    plt.show(block=False)
    plt.pause(15)
    plt.close(0)
