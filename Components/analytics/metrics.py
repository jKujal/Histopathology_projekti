import os

import pandas as pd

from Components.analytics import plots



def logging(args, logs_path, results_list, train_loss, val_loss, epoch, spsht_index):

    if epoch % 5 == 0:

        plots.create_loss_plot(train_loss, val_loss, logs_path, epoch, spsht_index, save=False, show=True)

    elif epoch == args.n_epoch:
        headings = {"Training loss", "Validation loss", "ground_truth",
                     "predictions:", "Epoch", "Split_#", "TP",
                     "TN", "FN", "FP"}
        results_list.insert(0, headings)
        results = pd.DataFrame(results_list)
        results.to_csv(os.path.join(logs_path, f"Results_epoch{epoch}_snapshot{spsht_index}"))

        plots.create_loss_plot(train_loss, val_loss, logs_path, epoch, spsht_index, save=True, show=True)

