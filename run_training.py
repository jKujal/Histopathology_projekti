import gc
import os
import time
import warnings
import random
import cv2
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import MultiStepLR
from tabulate import tabulate
from Components.data_processing import dataset_splits, histodatahandler
from Components.analytics import metrics
from Components.data_processing.dataset_splits import initiate_gfk_splits, init_folds, init_loaders, split_train_holdout
from Components.training import arguments, utilities
from torchinfo import summary

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":

    args = arguments.parse_args()

    # Lockdown seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_file_path = histodatahandler.create_histo_data()
    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    logs_path = os.path.join("/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo", "Logs", snapshot_name)
    os.makedirs(os.path.join(logs_path), exist_ok=True)

    metadata = pd.read_csv(data_file_path)

    train_metadata = split_train_holdout(args, metadata, os.path.join(logs_path, "Data"), dataset="histo")
    cv_splits = initiate_gfk_splits(args, train_metadata)
    cv_split_train_val = init_folds(args, cv_splits)

    net = utilities.init_model(args)
    optimizer = utilities.init_optimizer(args, net.parameters())

    if args.optimizer == 'sgd':
        print('Learning rate drop schedule: ', args.lr_drop)
        scheduler = MultiStepLR(optimizer, milestones=args.lr_drop,
                                gamma=args.learning_rate_decay)  # No need with adam(?)

    # Show structure and forward feed shapes


    summary(net, input_size=(args.batch_size, 3, 50, 50),
            col_names=['input_size', 'output_size', 'kernel_size', 'trainable'])

    nth_split = 1;
    # Setup confusion matrix fancy print
    labels = ["True", "False"]
    headers = [""] + labels

    snapshots = [None, None]

    for spsht_index, _ in enumerate(snapshots):

        # if index = 1:
        #     load pretrained weights and train again
        #
        for fold_id in cv_split_train_val:

            train_index, val_index = cv_split_train_val[fold_id]
            train_loader, val_loader = init_loaders(args, train_split=metadata.iloc[train_index],
                                                    val_split=metadata.iloc[val_index])

            results_list = []
            train_list = []
            val_list = []
            criterion = utilities.init_loss()

            for epoch in range(args.n_epochs):

                train_loss = utilities.train_epoch(args, net, optimizer, train_loader, criterion, epoch)
                validation_loss, predictions, ground_truth, accuracy, confusion_matrix = utilities.validate_epoch(net,
                                                                                                                  val_loader,
                                                                                                                  criterion,
                                                                                                                  args,
                                                                                                                  epoch)

                print('Validation accuracy: ', round(accuracy, 4))
                results_list.append(
                    {"Training loss": train_loss, "Validation loss": validation_loss, "ground_truth:": ground_truth,
                     "predictions:": predictions, "Epoch": epoch, "Split_#": nth_split, "TP": confusion_matrix[0, 0],
                     "TN": confusion_matrix[1, 1], "FN": confusion_matrix[0, 1], "FP": confusion_matrix[1, 0]})
                train_list.append(train_loss)
                val_list.append(validation_loss)

                # Print confusion matrix
                table = [[label] + row.tolist() for label, row in zip(labels, confusion_matrix)]
                print(tabulate(table, headers=headers, tablefmt="grid"))

                if epoch % 9 == 0 and epoch != 0:
                    metrics.logging(args, logs_path, results_list, train_list, val_list, epoch, spsht_index)

                if args.optimizer == 'sgd':
                    scheduler.step()
                gc.collect()

            nth_split += 1

        state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict()}
        cur_snapshot_name = os.path.join(logs_path, args.net + "_" + args.optimzer + "_" + f"split#{nth_split}.pth")
        print(cur_snapshot_name)
        torch.save(state, cur_snapshot_name)
        snapshots[spsht_index] = cur_snapshot_name

        torch.cuda.empty_cache()
        del net
