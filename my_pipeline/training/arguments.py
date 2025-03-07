import argparse
from pathlib import Path

import ast


def aslist(lst):
    return ast.literal_eval(lst)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', choices=['VGG', 'VGG_like'], default='VGG')
    parser.add_argument('--data_split', choices=['sss', 'sgkf', ], default='sss')
    parser.add_argument('--model', choices=['vgg_like', 'vgg16', ], default='vgg16')
    parser.add_argument('--seed', type=int, default='481516')
    parser.add_argument("--use_local_weights", type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=32)

    parser.add_argument('--k_folds', type=int, choices=[5, 10], default=5)
    parser.add_argument('--n_splits', type=int, default=1)
    parser.add_argument('--lr', type=float, choices=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10],
                        default=1e-2)  # Can increase
    parser.add_argument('--lr_drop', type=aslist, choices=[[160, 260]], default=[160, 260])
    parser.add_argument('--wd', type=float, choices=[5e-4, 1e-4, 1e-3, 1e-2], default=5e-4)
    parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], default='sgd')
    parser.add_argument('--set_nesterov', default=True)
    parser.add_argument('--learning_rate_decay', type=float, choices=[0.1, 0.2, ], default=0.2)
    parser.add_argument('--n_epochs', type=int, choices=[2, 10, 25, 50], default=25)  # max 50 for tests
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--transform', type=bool, default=False)
    parser.add_argument('--train_fold', type=int, choices=[-1], default=-1)

    args = parser.parse_args()
    return args
