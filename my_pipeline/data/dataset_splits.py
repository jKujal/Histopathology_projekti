import numpy as np
import pandas as pd
import torch
import solt.core as slc
import solt.transforms as slt
import cv2
from PIL import Image
from numpy.ma.core import concatenate
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from my_pipeline.data import transformations


class ImageDataset(Dataset):
    def __init__(self, split, transforms=None):
        self.dataframe = split
        self.transform = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        image_path = self.dataframe.iloc[idx, 2]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label = self.dataframe.iloc[idx, 1]
        group = self.dataframe.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'label': label, 'group': group}


def create_groupKfold_split(csv_file_path, args):
    '''
    Create StratifiedGroupKfold split from a given csv. file. Create csv files of the split
    :param filePath: Path to the .csv-file to be split
    :return: Path to created
    '''

    # CSV-file to Pandas
    df = pd.read_csv(csv_file_path)
    df_cancer = df.loc[df['Class'] == 1]  # subsampling
    df_normal = df.loc[df['Class'] == 0]

    df_cancer = df_cancer.sample(n=500, random_state=args.seed)
    df_normal = df_normal.sample(n=500, random_state=args.seed)
    df = pd.concat([df_cancer, df_normal]).reset_index(drop=True)

    # Instance of image transformation function
    transform = transformations.image_transformation()  # Do image manipulations

    validation_size = 0.1
    skgf = StratifiedGroupKFold(n_splits=args.k_folds)
    sss_test_and_val = StratifiedShuffleSplit(args.n_splits, test_size=validation_size, random_state=args.seed)

    loaders = []
    # Split creates indices for training and test data
    for train_index, test_valid_index in skgf.split(df, df['Class']):
        # From all the data, fetch training/test data based on created indices
        train_df = df.iloc[train_index]
        test_valid_df = df.iloc[test_valid_index]

        train_dataset = ImageDataset(train_df, transforms=transform)
        test_dataset = ImageDataset(test_valid_df, transforms=transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        loaders.append([train_loader, test_loader])

    i = 0;
    for test_index, valid_index in sss_test_and_val.split(df, df['Class']):
        validation_df = df.iloc[valid_index]

        validation_dataset = ImageDataset(validation_df, transforms=transform)

        validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

        loaders[i] += (validation_loader,)
        i += 1
    return loaders


def create_shuffle_split(csv_file_path, args):
    '''

    :param csv_file:
    :param args:
    :return:
    '''

    df = pd.read_csv(csv_file_path)
    df_cancer = df.loc[df['Class'] == 1]  # subsampling
    df_normal = df.loc[df['Class'] == 0]

    df_cancer = df_cancer.sample(n=1000, random_state=args.seed)
    df_normal = df_normal.sample(n=1000, random_state=args.seed)
    df = pd.concat([df_cancer, df_normal]).reset_index(drop=True)

    # Instance of image transformation function
    transform = transformations.image_transformation()

    # Standard ratio between training:validation:test = (0.6-0.8):(0.1-0.2):(0.1-0.2)
    args.n_splits = 5
    test_size = 0.1
    validation_size = 0.1
    training_size = 0.8
    test_val_size = 0.2
    sss_train_and_val = StratifiedShuffleSplit(args.n_splits, test_size=test_val_size, random_state=args.seed)

    loaders = []
    # Split creates indices for training and test data
    for train_index, test_valid_index in sss_train_and_val.split(df, df['Class']):
        # From all the data, fetch training/test data based on created indices
        train_df = df.iloc[train_index]
        test_valid_df = df.iloc[test_valid_index]

        # Do image transformations for each dataset
        train_dataset = ImageDataset(train_df, transforms=transform)

        # Pytorch dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Original test_val size = 20% of original data size, 10% each for test and validation sets means 50% from that last 20% needs to be put into the test_size here.
        sss_test_and_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)

        for test_index, valid_index in sss_test_and_val.split(test_valid_df, test_valid_df['Class']):
            test_df = test_valid_df.iloc[test_index]
            validation_df = test_valid_df.iloc[valid_index]

            test_dataset = ImageDataset(test_df, transforms=transform)
            validation_dataset = ImageDataset(validation_df, transforms=transform)

            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

            loaders.append([train_loader, test_loader, validation_loader])

    return loaders


def concatenate_column_values(dframe, cols):
    """
    @param dframe: Pandas DataFrame
    @param cols: List of columns
    @return: Pandas DataFrame
    """
    return pd.Series(map(''.join, dframe[cols].values.astype(str).tolist()), index=dframe.index)


def initiate_splits(args, metadata):
    gfk = StratifiedGroupKFold(n_splits=args.k_folds)

    y = concatenate_column_values(dframe=metadata, cols=['FolderID'])
    gfk_split = gfk.split(metadata, y=y, groups=metadata.FolderID.astype(str))

    cv_split = [x for x in gfk_split]

    return cv_split


def split_train_holdout(args, metadata, metadata_folder, dataset='histo'):
    gss = GroupShuffleSplit(n_splits=args.n_splits, test_size=0.2, train_size=0.8, random_state=args.seed)
    gss_split = gss.split(metadata, metadata.Class, metadata.FolderID)

    split = [x for x in gss_split]

    training_index = split[0][0]
    holdout_index = split[0][1]

    train_metadata = metadata.iloc[training_index]
    holdout_metadata = metadata.iloc[holdout_index]

    train_metadata.to_csv(f"{metadata_folder}/{dataset}_train_metadata.csv", index=None)
    holdout_metadata.to_csv(f"{metadata_folder}/{dataset}_holdout_metadata.csv", index=None)

    return split


def init_folds(args, cv_split):
    cv_split_train_val = {}

    for fold_id, split in enumerate(cv_split):
        if fold_id != args.train_fold and args.train_fold > -1:
            continue
        cv_split_train_val[fold_id] = split

    return cv_split_train_val


def init_loaders(args, train_split, val_split):
    transformations = init_transformations(args)

    if args.transform:
        train_dataset = ImageDataset(split=train_split, transforms=transformations['train'])
        val_dataset = ImageDataset(split=val_split)

    else:
        train_dataset = ImageDataset(split=train_split)
        val_dataset = ImageDataset(split=val_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, drop_last=True,
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.n_threads)

    return train_loader, val_loader


def init_transformations(args):
    rotation_range = (-5, 5)
    train_trsf = slc.Stream([
        slt.Flip(p=0.25, axis=-1),
        slt.Rotate(angle_range=(rotation_range[0], rotation_range[1]), p=0.25),
    ])

    return {"train": train_trsf}
