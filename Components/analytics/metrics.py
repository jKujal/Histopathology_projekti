import os

import pandas as pd

from Components.analytics import plots



def logging(args, logs_path, results_list, train_loss, val_loss, acc_list, epoch, fold_id, name):

    if epoch % 5 == 0 and epoch > 1:

        if not os.path.exists(os.path.join(logs_path, "Figures")):
            os.makedirs(os.path.join(logs_path, "Figures"))

        if epoch == args.n_epochs - 1:

            if not os.path.exists(os.path.join(logs_path, "Figures")):
                os.makedirs(os.path.join(logs_path, "Figures"))

            results = pd.DataFrame(results_list)
            results.to_csv(os.path.join(logs_path, f"{name}_results_fold{fold_id}_epoch{epoch}.csv"), index=False)

            plots.create_loss_plot(train_loss, val_loss, logs_path, fold_id, epoch, name, save=True, show=False)
            plots.create_acc_plot(logs_path, name, fold_id, epoch, acc_list, save=True, show=False)
        else:

            plots.create_loss_plot(train_loss, val_loss, logs_path, epoch, fold_id, name, save=True, show=False)
            plots.create_acc_plot(logs_path, name, fold_id, epoch, acc_list, save=True, show=False)
