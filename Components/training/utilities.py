import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from torch import optim
from torchvision.models import vgg16_bn
from tqdm import tqdm
from Components.models import vgg_types


def adjust_state_dict(state_dict, num_classes=2):
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'classifier.5' not in k:  # Exclude weights and biases of the last layer
            new_state_dict[k] = v
    return new_state_dict


def init_model(args, classes=10):
    if args.model == 'vgg16':
        net = vgg16_bn(weights=False)
        num_ftrs = net.classifier[6].in_features  # ?
        net.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=2, bias=True)  # ?
        net = net.to('cuda:0')
        return net
    elif args.model == 'VGGNDrop':
        net = vgg_types.VGGBNDrop(num_classes=classes, init_weights=False)
        net = net.to('cuda:0')
        return net
    elif args.model == 'VGG':

        net = vgg_types.VGG(num_classes=classes, init_weights=True)
        net = net.to('cuda:0')
        return net


def init_loss():
    criterion = nn.BCEWithLogitsLoss

    return criterion


def init_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        return optim.SGD(parameters, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=args.set_nesterov)
    else:
        raise NotImplementedError


def train_epoch(args, net, optimizer, train_loader, criterion, epoch, fold_id):
    net.train(True)
    running_loss = 0.0

    n_batches = len(train_loader)
    max_epochs = args.n_epochs

    # Grab "next" iterable from..
    device = next(net.parameters()).device

    progress = tqdm(total=n_batches)
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        # path = Path(batch['image_path'][0]).stem

        optimizer.zero_grad()

        # Forwards feed
        outputs = net(images)  #
        loss = criterion()(outputs.squeeze(), labels.float())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        progress.set_description(
            f'Fold {fold_id}, [{epoch + 1} | {max_epochs+1}] Average train loss: {running_loss / (i + 1):.3f} / Batch loss {loss.item():.3f}')  #
        progress.update()

        gc.collect()
        torch.cuda.empty_cache()

    progress.close()

    return running_loss / n_batches


def validate_epoch(net, test_loader, criterion, args, epoch):
    net.eval()
    running_loss = 0.0
    n_batches = len(test_loader)
    max_epoch = args.n_epochs
    device = next(net.parameters()).device
    predictions_list = []
    ground_truth_list = []

    progress = tqdm(total=n_batches)

    correct = 0
    all_samples = 0
    confusion_matrix = np.zeros((2, 2), dtype=np.uint64)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = net(images)

            loss = criterion()(outputs.squeeze(), labels.float())

            probs_batch = F.sigmoid(outputs).data.to('cpu').numpy()
            predictions = (probs_batch > 0.5).astype(int)
            predictions = np.array([item[0] for item in predictions])
            ground_truth_batch = batch['label'].numpy()

            predictions_list.extend(predictions.tolist())
            ground_truth_list.extend(ground_truth_batch.tolist())

            running_loss += loss.item()

            # Compare predicted labels to the actual true labels, and calculate correct=True predictions
            correct += np.equal(predictions_list, ground_truth_list).sum()

            confusion_matrix += calculate_confusion_matrix_from_arrays(np.array(predictions_list), np.array(ground_truth_list), 2)

            all_samples += len(np.array(ground_truth_list))

            progress.set_description(
                f'[{epoch + 1} | {max_epoch+1}] Validation accuracy: {100. * correct / all_samples:.04f}%')
            progress.update()

            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

    return running_loss / n_batches, np.array(predictions_list), np.array(
        ground_truth_list), correct / all_samples, confusion_matrix


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