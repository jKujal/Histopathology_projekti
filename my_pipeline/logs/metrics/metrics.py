from my_pipeline.logs.metrics import plots


def logs(args, logs_path, results_list, train_loss, val_loss, epoch):

    # plots.create_confusion_matrix()
    plots.create_loss_plot(train_loss, val_loss, logs_path, epoch)

