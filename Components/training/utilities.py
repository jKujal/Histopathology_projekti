import gc
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as c_matrix
from torch import nn
from torch import optim
from torchvision.models import vgg16_bn
from tqdm import tqdm
from Components.models import vgg_types

DEBUG = sys.gettrace() is not None

def adjust_state_dict(state_dict, num_classes=2):
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'classifier.5' not in k:  # Exclude weights and biases of the last layer
            new_state_dict[k] = v
    return new_state_dict


def init_model(args, classes=10, TSNE=False):
    if args.model == 'vgg16':
        net = vgg16_bn(weights=False)
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=2, bias=True)
        net = net.to('cuda:0')
        return net
    elif args.model == 'VGGNDrop':
        net = vgg_types.VGGBNDrop(num_classes=classes, init_weights=False, TSNE=TSNE)
        net = net.to('cuda:0')
        return net
    elif args.model == 'VGG':

        net = vgg_types.VGG(num_classes=classes, init_weights=True)
        net = net.to('cuda:0')
        return net

def init_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        return optim.SGD(parameters, lr=args.lr, weight_decay=args.wd, momentum=args.sgd_momentum, nesterov=args.set_nesterov)
    else:
        raise NotImplementedError


def train_epoch(args, net, optimizer, train_loader, epoch, fold_id, name):
    net.train(True)

    running_loss = 0.0

    n_batches = train_loader.batch_size
    max_epochs = args.n_epochs

    # Grab "next" iterable from..
    device = next(net.parameters()).device

    progress = tqdm(total=len(train_loader))
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        images = batch['image'].to(device).float()
        labels = batch['label'].to(device)
        # path = Path(batch['image_path'][0]).stem
        # Forwards feed
        outputs = net(images)

        pos_weight = torch.tensor(labels.shape) * 2.5
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))(outputs.squeeze(), labels.float())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        progress.set_description(
            f'{name} Fold {fold_id}, [{epoch + 1} | {max_epochs+1}] Average train loss: {running_loss / (i + 1):.3f} / Batch loss {loss.item():.3f}')  #
        progress.update()

        gc.collect()
        torch.cuda.empty_cache()

    progress.close()

    return running_loss / n_batches


def validate_epoch(net, val_loader, args, epoch):
    net.eval()
    running_loss = 0.0
    n_batches = val_loader.batch_size
    device = next(net.parameters()).device
    predictions_list = []
    ground_truth_list = []

    progress = tqdm(total=len(val_loader))

    correct = 0
    all_samples = 0
    running_f1 = 0.0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch['image'].to(device).float()
            labels = batch['label'].to(device)
            path = batch['image_path']
            outputs = net(images)
            pos_weight = torch.tensor(labels.shape) * 2.5
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))(outputs.squeeze(), labels.float())

            # if DEBUG:
            #     axes = plt.subplot(2, int(n_batches / 2), 1)
            #
            #     for k, path in enumerate(path):
            #
            #         image = cv2.imread(path)
            #         axes.flat[k].imshow(image)
            #         plt.title(f'Image # {k}')
            #         plt.show(block=False)
            #         plt.pause(5)
            #         plt.close()
            # probs_batch = F.sigmoid(outputs).data.to('cpu').numpy()

            probs = torch.sigmoid(outputs).squeeze()
            predictions = (probs > 0.5).int().to('cpu').numpy()

            predictions_list.extend(predictions.tolist())
            ground_truth_list.extend(labels.to('cpu').numpy().tolist())
            f1 = f1_score(np.array(ground_truth_list), np.array(predictions_list), average='binary', zero_division=0.0)
            running_f1 += f1
            running_loss += loss.item()

            confusion_matrix = c_matrix(np.array(ground_truth_list), np.array(predictions_list), labels=[0, 1])
            TP = confusion_matrix[0, 0]
            TN = confusion_matrix[1, 1]
            FN = confusion_matrix[0, 1]
            FP = confusion_matrix[1, 0]
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)

            # Compare predicted labels to the actual true labels, and calculate correct=True predictions
            correct = np.equal(predictions_list, ground_truth_list).sum()

            all_samples = len(np.array(ground_truth_list))

            progress.set_description(
                f'[{epoch + 1} | {args.n_epochs+1}] Validation F1-score: {100. * f1 / all_samples:.03f}%, Recall: {recall:.03f}, Precision: {precision:.03f}, Accuracy: {100. * correct / all_samples:.03f}%')
            progress.update()

            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

    return running_loss / n_batches, np.array(predictions_list), np.array(
        ground_truth_list), correct / all_samples


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    """Calculate confusion matrix from arrays

    https://github.com/ternaus/robot-surgery-segmentation/blob/master/validation.py#L77-L88

    :param prediction:
    :param ground_truth:
    :param nr_labels:
    :return:

    """
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T

    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )

    confusion_matrix = confusion_matrix.astype(np.uint64)

    return confusion_matrix

class BCEWithLogitsLoss2d(nn.Module):
    """Computationally stable version of 2D BCE loss

    """

    def __init__(self):
        super(BCEWithLogitsLoss2d, self).__init__()

        self.bce_loss = nn.BCEWithLogitsLoss(None, reduction='mean')

    def forward(self, logits, targets):
        logits_flat = logits.view(-1)
        targets_flat = targets.view(-1)
        return self.bce_loss(logits_flat, targets_flat)