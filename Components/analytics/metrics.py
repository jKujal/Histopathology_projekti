import os

import pandas as pd

from Components.analytics import plots



def logging(args, logs_path, results_list, train_loss, val_loss, acc_list, f1_list, epoch, fold_id, name):

    if epoch % args.save_interval == 0 and epoch > 1:

        if not os.path.exists(os.path.join(logs_path, "Figures")):
            os.makedirs(os.path.join(logs_path, "Figures"))

        if epoch == args.n_epochs:

            if not os.path.exists(os.path.join(logs_path, "Figures")):
                os.makedirs(os.path.join(logs_path, "Figures"))

            results = pd.DataFrame(results_list)
            results.to_csv(os.path.join(logs_path, f"{name}_results_fold{fold_id}_epoch{epoch}.csv"), index=False)

            plots.create_loss_plot(args, train_loss, val_loss, logs_path, fold_id, epoch, name, save=True, show=False)
            plots.create_acc_plot(args, logs_path, name, fold_id, epoch, acc_list, f1_list, save=True, show=False)
        else:

            plots.create_loss_plot(args, train_loss, val_loss, logs_path, epoch, fold_id, name, save=False, show=False)
            plots.create_acc_plot(args, logs_path, name, fold_id, epoch, acc_list, f1_list, save=False, show=False)
