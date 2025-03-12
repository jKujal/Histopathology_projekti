import os

import pandas as pd

from Components.analytics import plots



def logging(args, logs_path, results_list, train_loss, val_loss, epoch, fold_id, spsht_index):

    if epoch % 5 == 0 and epoch != 0:

        plots.create_loss_plot(train_loss, val_loss, logs_path, epoch, fold_id, spsht_index, save=False, show=True)

    elif epoch == args.n_epochs - 1:

        results = pd.DataFrame(results_list)
        results.to_csv(os.path.join(logs_path, f"Results_fold{fold_id}_epoch{epoch}_snapshot{spsht_index}"), index=False)

        plots.create_loss_plot(train_loss, val_loss, logs_path, epoch, spsht_index, save=True, show=True)

