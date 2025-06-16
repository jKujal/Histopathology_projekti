import gc
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from pathlib import Path

import torchvision.transforms.v2
from qhoptim.pyt import QHM, QHAdam
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix as c_matrix
from torch import nn
from torch import optim
from torchvision.models import vgg16_bn, resnet34, resnet50, ResNet34_Weights, ResNet50_Weights, vgg11
from tqdm import tqdm
from torchvision import transforms
from Components.models import vgg_types

DEBUG = sys.gettrace() is not None


def adjust_state_dict(state_dict, num_classes=1):
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'classifier.5' not in k:  # Exclude weights and biases of the last layer
            new_state_dict[k] = v
    return new_state_dict


def init_model(args, classes=2, TSNE=False):
    if args.model == 'vgg16':
        net = vgg16_bn(weights=False)
        num_ftrs = net.classifier[6].in_features
        net.classifier[6] = nn.Linear(in_features=num_ftrs, out_features=1, bias=True)
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
    elif args.model == 'my_test_net':

        net = vgg_types.my_test_net(num_classes=classes)
        net = net.to('cuda:0')
        return net
    elif args.model == 'resnet34':

        net = resnet34(weights=ResNet34_Weights)
        net.fc = nn.Linear(512, 1)
        net = net.to('cuda:0')
        return net
    elif args.model == 'resnet50':

        net = resnet50(weights=ResNet50_Weights)
        net.fc = nn.Linear(512 * 4, 1)
        net = net.to('cuda:0')

        return net
    elif args.model == 'this':
        net = torchvision.models.vgg16(pretrained=True)
        net.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
        net = net.to('cuda:0')

        return net
    elif args.model == 'vgg11':
        net = torchvision.models.vgg11()
        net.classifier[6] = nn.Linear(in_features=4096, out_features=2, bias=True)
        net = net.to('cuda:0')

        return net


def init_optimizer(args, parameters):
    if args.optimizer == 'adam':
        return optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == 'sgd':
        return optim.SGD(parameters, lr=args.lr, weight_decay=args.wd, momentum=args.sgd_momentum,
                         nesterov=args.set_nesterov)
    elif args.optimizer == 'qhoptim':
        return QHAdam(parameters, lr=args.lr, nus=(0.7, 1.0), betas=(0.995, 0.999))
    else:
        raise NotImplementedError


def train_epoch(args, net, optimizer, train_loader, epoch, fold_id, name):
    net.train(True)

    running_loss = 0.0
    n_batches = len(train_loader)
    max_epochs = args.n_epochs

    # Grab "next" iterable from.., used for multi-gpu setups
    device = next(net.parameters()).device
    loss_function = nn.CrossEntropyLoss()
    progress = tqdm(total=n_batches)

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        paths = batch['image_path']

        # if DEBUG:
        #     for i, image in enumerate(images):
        #
        #         fig = plt.figure()
        #         img = image.cpu().permute(1,2,0).numpy() # from torch tensor to correct shape numpy array
        #         plt.imshow(img)
        #         plt.title(f'{Path(paths[i]).stem}') # show image directory as a title
        #         plt.show(block=False)
        #         plt.pause(1)
        #         plt.close()

        outputs = net(images.float())

        # pos_weight = torch.ones(labels.shape).to('cuda:0') * 1.5 # This could be used to give more "value" to class 1 images
        # labels.to('cuda0')
        loss = loss_function(outputs.squeeze(), labels)
        # pos_weight=pos_weight.to(device)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        progress.set_description(
            f'{name} Fold {fold_id}, [{epoch + 1} | {max_epochs + 1}] Average train loss: {running_loss / (i + 1):.3f} / Batch loss {loss.item():.3f}')  #
        progress.update()

        gc.collect()
        torch.cuda.empty_cache()

    progress.close()

    return running_loss / n_batches


def validate_epoch(net, val_loader, args, epoch):
    net.train(False)  # model on evaluation mode, no updates are done to the weights!
    running_loss = 0.0
    n_batches = len(val_loader)
    device = next(net.parameters()).device
    predictions_list = []
    ground_truth_list = []

    progress = tqdm(total=len(val_loader))

    correct = 0
    all_samples = 0
    f1 = 0

    loss_function = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            paths = batch['image_path']

            # if DEBUG:
            #     for i, image in enumerate(images):
            #         fig = plt.figure()
            #         img = image.cpu().permute(1, 2, 0).numpy() # from torch tensor to correct shape numpy array
            #         plt.imshow(img)
            #         plt.title(f'{Path(paths[i]).stem}') # show image directory as a title
            #         plt.show(block=False)
            #         plt.pause(1)
            #         plt.close()

            outputs = net(images.float())

            # pos_weight = torch.ones(labels.shape).to('cuda:0') * 1.5 # This could be used to give more "value" to class 1 images
            # labels.to('cuda0')
            loss = loss_function(outputs.squeeze(), labels)  # pos_weight=pos_weight.to(device)

            _, predictions = torch.max(outputs.data, 1)

            predictions_list.extend(predictions.cpu())
            ground_truth_list.extend(labels.to('cpu').numpy().tolist())

            running_loss += loss.item()

            correct += (predictions == labels).sum().item()

            all_samples = len(np.array(ground_truth_list))
            if i + 1 == len(val_loader):
                confusion_matrix = c_matrix(np.array(ground_truth_list), np.array(predictions_list), labels=[0, 1])
                TP = confusion_matrix[0, 0]
                TN = confusion_matrix[1, 1]
                FN = confusion_matrix[0, 1]
                FP = confusion_matrix[1, 0]
                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                f1 = f1_score(np.array(ground_truth_list), np.array(predictions_list), average='binary',
                              zero_division='warn')

                progress.set_description(
                    f'[{epoch + 1} | {args.n_epochs + 1}] Validation F1-score: {100. * (f1):.03f}%, Recall: {recall:.03f}, Precision: {precision:.03f}, Accuracy: {100. * correct / all_samples:.03f}%, Average Val Loss: {running_loss / i}')
                progress.update()
            else:
                progress.set_description(
                    f'[{epoch + 1} | {args.n_epochs + 1}] Accuracy: {100. * correct / all_samples:.03f}%, Loss: {loss}')
                progress.update()
            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

    return running_loss / n_batches, np.array(predictions_list), np.array(ground_truth_list), correct / all_samples, f1
