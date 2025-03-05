import os
from my_pipeline.logs.metrics import plots



def logs(args, logs_path, results_list, train_loss, val_loss, epoch, index):

    os.makedirs(os.path.join(logs_path), exist_ok=True)

    # plots.create_confusion_matrix()
    plots.create_loss_plot(train_loss, val_loss, logs_path, epoch, index)

