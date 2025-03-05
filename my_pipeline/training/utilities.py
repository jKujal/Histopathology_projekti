import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision.models import vgg16_bn
from tqdm import tqdm
from my_pipeline.models import vgg_like


def init_model(args):
    if args.model == 'vgg16':
        net = vgg16_bn(weights=False)
        num_ftrs = net.classifier[6].in_features  # ?
        net.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=2, bias=True)  # ?
        net = net.to('cuda:0')
        return net
    else:
        net = vgg_like.VGG(num_classes=2, init_weights=True)
        net = net.to('cuda:0')
        return net


def init_loss():
    criterion = nn.CrossEntropyLoss()

    return criterion


def init_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        return optim.SGD(parameters, lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=args.set_nesterov)
    else:
        raise NotImplementedError


def train_epoch(args, net, optimizer, train_loader, criterion, epoch):
    net.train(True)
    running_loss = 0.0

    n_batches = len(train_loader)
    max_epochs = args.n_epochs

    # Grab "next" iterable from..
    device = next(net.parameters()).device

    progress = tqdm(total=n_batches)
    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        labels = batch['label'].long().to(device)

        optimizer.zero_grad()

        # Forwards feed
        outputs = net(images)  #
        loss = criterion(outputs, labels)  # CrossEntropyLoss

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        progress.set_description(
            f'[{epoch + 1} | {max_epochs}] Train loss: {running_loss / (i + 1):.3f} / Loss {loss.item():.3f}')  #
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

    probs_lst = []
    ground_truth_list = []

    progress = tqdm(total=n_batches)

    correct = 0
    all_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            labels = batch['label'].long().to(device)

            outputs = net(images)

            # Cross entropy loss model output vs actual labels for batch
            # Forwards feed
            # F.binary_cross_entropy?
            loss = criterion(outputs, labels)

            # Squish the output values to a probability range between 0 and 1
            # In our case because there are only 2 classes..
            probs_batch = F.softmax(outputs, 1).data.to('cpu').numpy()

            ground_truth_batch = batch['label'].numpy()

            probs_lst.extend(probs_batch.tolist())
            ground_truth_list.extend(ground_truth_batch.tolist())

            running_loss += loss.item()

            # Get the predicted class, which is the one with the highest probability, the maximum value
            prediction = np.array(probs_lst).argmax(1)
            # Compare predicted labels to the actual true labels, and calculate correct=True predictions
            correct += np.equal(prediction, np.array(ground_truth_list)).sum()

            all_samples += len(np.array(ground_truth_list))

            progress.set_description(
                f'[{epoch + 1} | {max_epoch}] Validation accuracy: {100. * correct / all_samples:.0f}%')
            progress.update()

            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

    return running_loss / n_batches, np.array(probs_lst), np.array(ground_truth_list), correct / all_samples
