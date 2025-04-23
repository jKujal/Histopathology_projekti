import numpy as np
import pandas as pd
import solt.constants
import torch
import solt.core as slc
import solt.transforms as slt
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit, GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tv_transforms

class ImageDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        entry = self.dataset.iloc[idx]
        image_path = entry.Image_path
        label = entry.Class
        folderid = entry.FolderID

        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = image.astype('uint8')

        if self.transforms is not None:

            dc_res = self.transforms({'image': image}, return_torch=False)

            transformed_img = dc_res.data[0].transpose((2, 0, 1)) / 255.0
            transformed_img = torch.from_numpy(transformed_img)
            transformed_img = tv_transforms.Normalize(mean=[0.73341537, 0.6338761, 0.8134402], std=[0.15178116, 0.21050893, 0.14175205] )(transformed_img.float())

            return {'image': transformed_img, 'label': label, 'image_path': image_path}
        else:
            image = np.array(image)
            image = image.transpose((2, 0, 1)) # numpy image: H x W x C, torch image: C x H x W
            image = torch.from_numpy(image)

            return {'image': image, 'label': label, 'folderid': folderid, 'image_path': image_path}


def concatenate_column_values(dframe, cols):
    """
    @param dframe: Pandas DataFrame
    @param cols: List of columns
    @return: Pandas DataFrame
    """
    return pd.Series(map(''.join, dframe[cols].values.astype(str).tolist()), index=dframe.index)


def initiate_sgkf_splits(args, metadata):

    sgkf = StratifiedGroupKFold(n_splits=args.k_folds)

    y = concatenate_column_values(dframe=metadata, cols=['Class'])
    sgkf_split = sgkf.split(metadata, y=metadata['Class'], groups=metadata.FolderID.astype(str))

    cv_split = [x for x in sgkf_split]

    return cv_split


def equal_sample(args, metadata, metadata_folder):

    cancer = metadata[metadata['Class'] == 1]
    not_cancer = metadata[metadata['Class'] == 0]

    sampled_cancer = cancer.sample(n=10000, random_state=args.seed)
    sampled_not_cancer = not_cancer.sample(n=10000, random_state=args.seed)

    metadata = pd.concat([sampled_cancer, sampled_not_cancer])

    gss = GroupShuffleSplit(n_splits=args.n_splits, test_size=0.2, train_size=0.8, random_state=args.seed)
    gss_split = gss.split(metadata, metadata.Class, metadata.FolderID)

    split = [x for x in gss_split]

    training_index = split[0][0]
    holdout_index = split[0][1]

    train_metadata = metadata.iloc[training_index]
    holdout_metadata = metadata.iloc[holdout_index]

    train_metadata.to_csv(os.path.join(metadata_folder, "histo_train_metadata.csv"), index=None)
    holdout_metadata.to_csv(os.path.join(metadata_folder, "histo_holdout_metadata.csv"), index=None)

    return train_metadata

def split_train_holdout(args, metadata, metadata_folder, dataset='histo'):

    if args.subsample:
        metadata = metadata.groupby('Class').apply(
            lambda x: x.sample(frac=0.9)
        )

    gss = GroupShuffleSplit(n_splits=args.n_splits, test_size=0.2, train_size=0.8, random_state=args.seed)
    gss_split = gss.split(metadata, metadata.Class, metadata.FolderID)

    split = [x for x in gss_split]

    training_index = split[0][0]
    holdout_index = split[0][1]

    train_metadata = metadata.iloc[training_index]
    holdout_metadata = metadata.iloc[holdout_index]

    train_metadata.to_csv(os.path.join(metadata_folder, f"{dataset}_train_metadata.csv"), index=None)
    holdout_metadata.to_csv(os.path.join(metadata_folder,f"{dataset}_holdout_metadata.csv"), index=None)

    return train_metadata

#
# def test_split(args, metadata, metadata_folder):
#
#     set = metadata.read_csv()
#
#
#
#
#


# def init_folds(args, cv_split):
#     cv_split_train_val = {}
#
#     for fold_id, split in enumerate(cv_split):
#         if fold_id != args.train_fold and args.train_fold > -1:
#             continue
#         cv_split_train_val[fold_id] = split
#
#     return cv_split_train_val


def init_loaders(args, train_split, val_split):
    transformations = init_transformations(args)

    if args.transform:
        train_dataset = ImageDataset(dataset=train_split, transforms=transformations['train'])
        val_dataset = ImageDataset(dataset=val_split)

    else:
        train_dataset = ImageDataset(dataset=train_split)
        val_dataset = ImageDataset(dataset=val_split)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_threads, drop_last=True,
                              worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.n_threads)

    return train_loader, val_loader


# def wrap_solt(inp):
#     """
#
#     :param inp:
#     :return:
#     """
#     return slc.DataContainer(inp, 'IM', transform_settings={0: {'interpolation': 'bilinear'}, 1: {'interpolation': 'nearest'}})

def init_transformations(args):

    #args here if you want to alter these.
    # rotation_range = (-90, 90)
    # translation_range = 20

    train_trsf = slc.Stream([
        slt.Flip(p=0.25, axis=-1), # axis -1 means all axes
        slt.Rotate90(k=1, p=0.25),
        # slt.Rotate(angle_range=(rotation_range[0], rotation_range[1]), p=0.25),
        # slt.Translate(range_x=translation_range, range_y=translation_range, p=0.1),
        # slt.Noise(p=0.25, gain_range=0.1, data_indices=None),
        slt.Pad(pad_to=(50, 50), padding='r'),
        slt.Crop(crop_mode='r', crop_to=(50, 50)),
    ])

    return {"train": train_trsf}
