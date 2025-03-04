from my_pipeline.logs.metrics import plots


def logs(train_losses, validation_losses, ground_truth, predictions, args, log_path, nth_split):
    # Create the confusion matrix and loss plots

    plots.create_confusion_matrix(ground_truth, predictions, log_path, nth_split)
    plots.create_loss_plot(train_losses, validation_losses, log_path, nth_split)
