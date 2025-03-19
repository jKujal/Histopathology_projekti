import numpy as np
import pandas as pd
import torch
import solt.core as slc
import solt.transforms as slt
import cv2
import os
from PIL import Image
from numpy.ma.core import concatenate
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, GroupShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from Components.data_processing import transformations


class ImageDataset(Dataset):
    def __init__(self, split, transforms=None):
        self.dataframe = split
        self.transform = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        entry = self.dataframe.iloc[idx]
        image_path = entry.Image_path
        label = entry.Class
        folderid = entry.FolderID

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = image.astype('uint8')

        if self.transform:
            image = self.transform(image)
            return ""
        else:
            image = np.array(image).astype(np.float32)
            image = torch.from_numpy(image)
            image = image.transpose(0, 2)

            return {'image': image, 'label': label, 'folderid': folderid, 'image_path': image_path}


def concatenate_column_values(dframe, cols):
    """
    @param dframe: Pandas DataFrame
    @param cols: List of columns
    @return: Pandas DataFrame
    """
    return pd.Series(map(''.join, dframe[cols].values.astype(str).tolist()), index=dframe.index)


def split_train_holdout(args, metadata, metadata_folder, dataset='histo'):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=args.seed)
    gss_split = gss.split(metadata, metadata.ID, metadata.FolderID)

    split = [x for x in gss_split]

    training_idx = split[0][0]
    holdout_idx = split[0][1]

    train_metadata = metadata.iloc[training_idx]
    holdout_metadata = metadata.iloc[holdout_idx]

    train_metadata.to_csv(f'{metadata_folder}/{dataset}_train_metadata.csv', index=None)
    holdout_metadata.to_csv(f'{metadata_folder}/{dataset}_holdout_metadata.csv', index=None)

    return split

def initiate_sgkf_splits(args, metadata):

    sgkf = StratifiedGroupKFold(n_splits=args.k_folds)

    y = concatenate_column_values(dframe=metadata, cols=['FolderID'])
    sgkf_split = sgkf.split(metadata, y=y, groups=metadata.FolderID.astype(str))

    cv_split = [x for x in sgkf_split]

    return cv_split


def split_train_holdout(args, metadata, metadata_folder, dataset='histo'):

    if args.subsample:
        metadata = metadata.groupby('Class').apply(
            lambda x: x.sample(frac=0.005)
        )

    gss = GroupShuffleSplit(n_splits=args.n_splits, test_size=0.2, train_size=0.8, random_state=args.seed)
    gss_split = gss.split(metadata, metadata.Class, metadata.FolderID)

    split = [x for x in gss_split]

    training_index = split[0][0]
    holdout_index = split[0][1]

    train_metadata = metadata.iloc[training_index]
    holdout_metadata = metadata.iloc[holdout_index]

    os.makedirs(metadata_folder, exist_ok=True)
    train_metadata.to_csv(os.path.join(metadata_folder, f"{dataset}_train_metadata.csv"), index=None)
    holdout_metadata.to_csv(os.path.join(metadata_folder,f"{dataset}_holdout_metadata.csv"), index=None)

    return train_metadata


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
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.n_threads, pin_memory=True)

    return train_loader, val_loader


def init_transformations(args):
    rotation_range = (-5, 5)
    train_trsf = slc.Stream([
        slt.Flip(p=0.25, axis=-1),
        slt.Rotate(angle_range=(rotation_range[0], rotation_range[1]), p=0.25),
    ])

    return {"train": train_trsf}
