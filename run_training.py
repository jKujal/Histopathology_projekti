import gc
import os
import time
import warnings

import cv2
import torch
from torch.optim.lr_scheduler import MultiStepLR

from my_pipeline.data import dataset_splits
from my_pipeline.data import histodatahandler
from my_pipeline.logs.metrics import metrics
from my_pipeline.training import arguments
from my_pipeline.training import utilities

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

if __name__ == "__main__":

    # Fetch run parameters
    args = arguments.parse_args()

    # Create Histopathology image csv-files, if not already created. Fetch data files
    data_file_path = histodatahandler.create_histo_data()

    # Create data split
    if args.data_split == 'sgkf':
        loaders = dataset_splits.create_groupKfold_split(data_file_path, args)
    elif args.data_split == 'sss':
        loaders = dataset_splits.create_shuffle_split(data_file_path, args)
    else:
        raise NotImplementedError("Not implemented")

    # Fetch model
    net = utilities.init_model(args)

    # Show NN structure
    print(net)

    # Create directory for logs
    # snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    # log_path = os.path.join(args.dataset_name, snapshot_name)
    # os.makedirs(os.path.join(log_path), exist_ok=True)


    # Do training n_split times
    nth_split = 1;

    for train_loader, test_loader, validation_loader in loaders:

        net = utilities.init_model(args)

        optimizer = utilities.init_optimizer(args, net.parameters())

        if args.optimizer == 'sgd':
            print('Learning rate drop schedule: ', args.lr_drop)
            scheduler = MultiStepLR(optimizer, milestones=args.lr_drop,
                                    gamma=args.learning_rate_decay)  # No need with adam

        # List for keeping track of loss/cost
        train_losses = []
        # List for keeping track of validation loss
        val_losses = []

        criterion = utilities.init_loss()

        for epoch in range(args.n_epochs):

            # print(colored('====> ', 'red') + 'Learning rate: ', str(scheduler.get_lr())[1:-1])
            train_loss = utilities.train_epoch(net, optimizer, train_loader, criterion, args, epoch)
            validation_loss, predictions, ground_truth, validation_accuracy = utilities.validate_epoch(net,
                                                                                                       validation_loader,
                                                                                                       criterion, args,
                                                                                                       epoch)

            print('Validation Accuracy: ', validation_accuracy)

            if args.optimizer == 'sgd':
                scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(validation_loss)

            if epoch % 9 == 0 and epoch != 0:
                metrics.logs(train_losses, val_losses, ground_truth, predictions, args, log_path, nth_split)

            # Free memory
            gc.collect()

        nth_split += 1

        # Free memory
        torch.cuda.empty_cache()
        del net
        gc.collect()
