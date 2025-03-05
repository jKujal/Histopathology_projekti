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

    args = arguments.parse_args()

    data_file_path = histodatahandler.create_histo_data()

    if args.data_split == 'sgkf':
        loaders = dataset_splits.create_groupKfold_split(data_file_path, args)
    elif args.data_split == 'sss':
        loaders = dataset_splits.create_shuffle_split(data_file_path, args)
    else:
        raise NotImplementedError("Not implemented")

    net = utilities.init_model(args)

    print(net)

    snapshot_name = time.strftime('%Y_%m_%d_%H_%M')
    logs_path = os.path.join("/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo", "Logs", snapshot_name)
    os.makedirs(os.path.join(logs_path), exist_ok=True)

    nth_split = 1;

    for train_loader, test_loader, val_loader in loaders:

        net = utilities.init_model(args)
        optimizer = utilities.init_optimizer(args, net.parameters())

        if args.optimizer == 'sgd':
            print('Learning rate drop schedule: ', args.lr_drop)
            scheduler = MultiStepLR(optimizer, milestones=args.lr_drop,
                                    gamma=args.learning_rate_decay)  # No need with adam(?)

        results_list = []
        train_list = []
        val_list = []
        criterion = utilities.init_loss()

        for epoch in range(args.n_epochs):

            train_loss = utilities.train_epoch(args, net, optimizer, train_loader, criterion, epoch)
            validation_loss, predictions, ground_truth, accuracy = utilities.validate_epoch(net,
                                                                                            val_loader,
                                                                                            criterion, args,
                                                                                            epoch)

            print('Validation accuracy: ', round(accuracy, 4))
            results_list.append({"Training loss": train_loss, "Validation loss": validation_loss, "ground_truth:": ground_truth, "predictions:": predictions, "Nth_split": nth_split})
            train_list.append(train_loss)
            val_list.append(validation_loss)

            if epoch % 9 == 0 and epoch != 0:
                metrics.logs(args, logs_path, results_list, train_list, val_list, epoch)

            if args.optimizer == 'sgd':
                scheduler.step()
            gc.collect()

        nth_split += 1

        torch.cuda.empty_cache()
        del net

