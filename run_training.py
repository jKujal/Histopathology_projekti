import gc
import os
import time
import warnings
import random
import cv2
import pandas as pd
import torch
import json
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix as c_matrix
from torch.optim.lr_scheduler import MultiStepLR
from tabulate import tabulate
from Components.data_processing import dataset_splits, histodatahandler
from Components.analytics import metrics
from Components.data_processing.dataset_splits import initiate_sgkf_splits, init_loaders, equal_sample
from Components.training import arguments, utilities
from torchinfo import summary

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

warnings.filterwarnings("ignore", category=UserWarning) # supress warnings that do not require fixing
warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":

    start_time = time.strftime('%Y_%m_%d_%H_%M_%S')

    # Fetch session arguments, including model training parameters, which model to use, etc.

    args = arguments.parse_args()

    # Lockdown seed to prevent randomness between each training session.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Make save directories

    data_file_path = histodatahandler.create_histo_data()
    snapshot_name = time.strftime(
        '%Y_%m_%d_%H_%M_%S') + f"_{args.optimizer}_epochs{args.n_epochs}_lr{args.lr}_lr_drop{str(args.untrained_lr_drop)}"
    logs_path = os.path.join("/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo", "Logs", snapshot_name)
    os.makedirs(os.path.join(logs_path), exist_ok=True)

    metadata = pd.read_csv(data_file_path)
    metadata_dir = os.path.join(logs_path, "Data")
    os.makedirs(metadata_dir, exist_ok=True)

    # Save args as a json, easy to read what parameters were used for this session

    args_dir = os.path.join(metadata_dir, 'training_arguments.txt')
    with open(args_dir, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Split the dataset first into Training / Testing sets, then split training set in to 5 folds of training / validation sets. cv_splits holds the indices from

    train_metadata = equal_sample(args, metadata, metadata_dir)
    cv_splits = initiate_sgkf_splits(args, train_metadata)

    # # Show model structure and forward feed shapes
    # summary(net, input_size=(args.batch_size, 3, 50, 50),
    #         col_names=['input_size', 'output_size', 'kernel_size', 'trainable'])

    # Setup confusion matrix fancy print
    labels = ["1", "0"]
    headers = [""] + labels

    runs = ['Random init', 'Pretrained'] # iterable for which models weights to use, random initialized weights vs. pretrained weights
    for index, name in enumerate(runs): # For each model type, train n = folds number of models
        fold_results = []

        for fold_id, content in enumerate(cv_splits): # Train a model for each fold

            args = arguments.parse_args()
            net = utilities.init_model(args, classes=2, TSNE=False)

            if name == 'Pretrained':

                args.lr = 0.001 # Models with pretrained weights need to be trained with different parameters, so that they do not overfit
                # args.wd = 0.01

                net_state_dict = torch.load(args.pretrained_model_params_dir, map_location='cpu', weights_only=False) # Load the pretrained weights as a python dictionary
                net_state = utilities.adjust_state_dict(net_state_dict, 2) # Change some aspects of the dictionary to match this binary classification task

                net.load_state_dict(net_state, strict=False) # Load the weights into the model

                optimizer = utilities.init_optimizer(args, net.parameters()) # Set the method for updating model weights

                if args.optimizer == 'sgd':
                    scheduler = MultiStepLR(optimizer, milestones=args.pretrained_lr_drop,
                                            gamma=args.learning_rate_decay) # optimizer specific settings
            else:

                optimizer = utilities.init_optimizer(args, net.parameters())

                if args.optimizer == 'sgd':
                    scheduler = MultiStepLR(optimizer, milestones=args.untrained_lr_drop,  # Note! .untrained, different learning rate / step length drop
                                            gamma=args.learning_rate_decay)

            # Fetch training / validation folds from the datasets

            train_index = content[0]
            val_index = content[1]
            train_data = train_metadata.iloc[train_index]
            val_data = train_metadata.iloc[val_index]
            fold_train_save = train_data.to_csv(os.path.join(metadata_dir, f"fold{fold_id}_training_data.csv")) # save the datasets as csv for debugging
            fold_val_save = val_data.to_csv(os.path.join(metadata_dir, f"fold{fold_id}_val_data.csv"))

            train_loader, val_loader = init_loaders(args, train_split=train_data,
                                                    val_split=val_data) # PyTorch based DataLoader for inputting the data to the model

            results_list = []
            train_list = []
            val_list = []
            acc_list = []
            f1_list = []
            best_f1 = 0

            for epoch in range(args.n_epochs + 1): # Actually train the model. Validate, i.e. check model performance on validation data after each training epoch

                train_loss = utilities.train_epoch(args, net, optimizer, train_loader, epoch, fold_id, name)
                validation_loss, predictions, ground_truth, accuracy, f1 = utilities.validate_epoch(net, val_loader,
                                                                                                    args, epoch)
                confusion_matrix = c_matrix(ground_truth, predictions)
                results_list.append(
                    {"Training loss": train_loss, "Validation loss": validation_loss, "Val_accuracy": accuracy,
                     "F1-score": f1, "ground_truth": ground_truth,
                     "predictions": predictions, "Epoch": epoch, "Fold_id": fold_id, "TP": confusion_matrix[0, 0],
                     "TN": confusion_matrix[1, 1], "FN": confusion_matrix[0, 1], "FP": confusion_matrix[1, 0],
                     "Confusion_matrix": confusion_matrix}) # Save metrics from epoch
                train_list.append(train_loss)
                val_list.append(validation_loss)
                acc_list.append(accuracy)
                f1_list.append(f1)

                # Print confusion matrix to terminal
                table = [[label] + row.tolist() for label, row in zip(labels, confusion_matrix)]
                print(tabulate(table, headers=headers, tablefmt="grid"))

                metrics.logging(args, logs_path, results_list, train_list, val_list, acc_list, f1_list, epoch, fold_id,
                                name) # Create plots from metrics

                if args.optimizer == 'sgd':
                    scheduler.step()

                if f1 > best_f1: # If the model in this epoch had better performance on the validation set, save the model "state"
                    best_f1 = f1
                    state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict()}
                    cur_snapshot_name = os.path.join(logs_path,
                                                     args.model + "_" + args.optimizer + "_" + f"best_{name}_weights_fold{fold_id}_@epoch{epoch}" + ".pth")
                    torch.save(state, cur_snapshot_name)
                    print(f"New best F1-score: {f1}")

                gc.collect()

            fold_results.append(results_list)
            torch.cuda.empty_cache() # Attempt to save GPU memory
            gc.collect()

            del net # Delete the model from cuda memory before starting to train another model

    end_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    start_time = datetime.strptime(start_time, '%Y_%m_%d_%H_%M_%S')
    end_time = datetime.strptime(end_time, '%Y_%m_%d_%H_%M_%S')
    duration = end_time - start_time
    print(f"Training ended after {duration}.")
