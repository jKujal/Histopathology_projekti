import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.lines import Line2D

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

    plt.plot(train_loss, linestyle='solid', label="Training loss")
    plt.plot(val_loss, linestyle='solid', label="Validation loss")
    plt.xticks(range(0, args.n_epochs + 1))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    # plt.autoscale(enable=True, axis="Both")
    plt.title(f"{name}, Training vs. Validation loss, lr: {args.lr}, wd: {args.wd}, optimizer: {args.optimizer}")
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
    plt.plot(acc_list, linestyle='solid', label='Accuracy')
    plt.plot(f1_list, linestyle='solid', label="F1-score")
    plt.xticks(range(0, args.n_epochs+1))
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    # plt.autoscale(enable=True, axis='Both')
    plt.legend()
    plt.title(f'{name}, F1-Score and accuracy on validation dataset, lr: {args.lr}, optimizer: {args.optimizer}')
    if save:
        plt.savefig(os.path.join(logs_path, "Figures", f"{name}_accuracy_fold{foldid}_epoch{epoch}"))
    if show:
        plt.show(block=False)
        plt.pause(5)
        plt.close(0)

def combined_plot(logs_path, pd_column_name: str):

    plot_data = []
    for file in os.listdir(logs_path):

        if "result" in file:
            fold_data = pd.read_csv(os.path.join(logs_path, file))

            if pd_column_name in fold_data.columns:

                if "Pretrained" in file:
                    fold_data.insert(0, "Dataset name", 'Pretrained')
                    plot_data.append(fold_data)
                else:
                    fold_data.insert(0, 'Dataset name', 'Untrained')
                    plot_data.append(fold_data)
            else:
                print("not defined")
                exit()


    plt.figure(figsize=(10, 5))

    pre_ave = []
    un_ave = []
    for data in plot_data:

        if data['Dataset name'][0] == 'Pretrained':
            pre_ave.append(data[pd_column_name])
            plt.plot(data[pd_column_name].values, '-', color='k', alpha=0.2)

        else:
            un_ave.append(data[pd_column_name])

            plt.plot(data[pd_column_name].values, '-', color='b', alpha=0.2)

    this = pd.DataFrame(pre_ave).transpose()
    this['mean'] = this.mean(axis=1)
    that = pd.DataFrame(un_ave).transpose()
    that['mean'] = that.mean(axis=1)

    plt.plot(this['mean'].values, '-', color='k')
    plt.plot(that['mean'].values, '-', color='b')
    # plt.yticks(range(0,110,10))
    plt.xticks(range(0, 21))
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.title(f'Validation loss, Pretrained folds vs. Untrained folds')
    legend_elements = [Line2D([0], [0], color='k', alpha=0.2, label='Pretrained folds'),
                       Line2D([0], [0], color='b', alpha=0.2, label='Untrained folds'),
                       Line2D([0], [0], color='k', label='Pretrained average'),
                       Line2D([0], [0], color='b',label='Untrained average')]
    plt.legend(handles=legend_elements, loc='best')

    plt.savefig(os.path.join(logs_path,f"{pd_column_name}_plot"))
    plt.show()

def combined_combined_plot(logs_path):

    plot_data = []
    for file in os.listdir(logs_path):

        if "result" in file:
            fold_data = pd.read_csv(os.path.join(logs_path, file))

            if "Pretrained" in file:
                fold_data.insert(0, "Dataset name", 'Pretrained')
                plot_data.append(fold_data)
            else:
                fold_data.insert(0, 'Dataset name', 'Untrained')
                plot_data.append(fold_data)

    plt.figure(figsize=(10, 5))

    trainloss_ave_pre = []
    valloss_ave_pre = []

    trainloss_ave_un = []
    valloss_ave_un = []

    for data in plot_data:

        if data['Dataset name'][0] == 'Pretrained':
            trainloss_ave_pre.append(data['Training loss'])
            plt.plot(data['Training loss'], '-', color='y', alpha=0.1)
            valloss_ave_pre.append(data['Validation loss'])
            plt.plot(data['Validation loss'], '-', color='y',alpha=0.1)
        else:
            trainloss_ave_un.append(data['Training loss'])
            plt.plot(data['Training loss'], '-', color='r', alpha=0.1)
            valloss_ave_un.append(data['Validation loss'])
            plt.plot(data['Validation loss'], '-', color='r', alpha=0.1)

    this = pd.DataFrame(trainloss_ave_pre).transpose()
    this_too = pd.DataFrame(valloss_ave_pre).transpose()

    this['mean'] = this.mean(axis=1)
    this_too['mean'] = this_too.mean(axis=1)

    that = pd.DataFrame(trainloss_ave_un).transpose()
    that_too = pd.DataFrame(valloss_ave_un).transpose()
    that['mean'] = that.mean(axis=1)
    that_too['mean'] = that_too.mean(axis=1)

    plt.plot(this['mean'].values, '-', color='k')
    plt.plot(this_too['mean'].values, '-', color='k', alpha=0.5)
    plt.plot(that['mean'].values, '-', color='b')
    plt.plot(that_too['mean'].values, '-', color='b', alpha=0.5)

    plt.xticks(range(0, 21))
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    plt.title(f'Training vs. Validation losses , Pretrained folds vs. Untrained folds')
    legend_elements = [Line2D([0], [0], color='k', alpha=1, label='Pretrained average training loss'),
                       Line2D([0], [0], color='b', alpha=1, label='Untrained average training loss'),
                       Line2D([0], [0], color='k', alpha=0.5,label='Pretrained average validation loss'),
                       Line2D([0], [0], color='b',alpha=0.5, label='Untrained average validation loss'),
                       Line2D([0], [0], color='y', alpha=0.1, label='Pretrained folds'),
                       Line2D([0], [0], color='r', alpha=0.1, label='Untrained folds')]
    plt.legend(handles=legend_elements, loc='best')

    plt.savefig(os.path.join(logs_path,'Losses_plot'))
    plt.show()