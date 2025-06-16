import os
import gc
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import solt.core as slc
import solt.transforms as slt
from argparse import ArgumentParser

from cv2.detail import resultRoi
from pathlib import Path
from faker import Faker
from Components.evaluation import tsne_tools
from Components.training import utilities
from tabulate import tabulate
from Components.data_processing.dataset_splits import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix as c_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from tqdm import tqdm

def run_inference(args_path, holdout_set, saved_state, name='This_result', show=False):
    """
    Run inference for a model trained for a binary classification problem
    Creates TSNE visualization for model output
    """
    #Load args
    parser = ArgumentParser()
    args = parser.parse_args()
    with open(args_path, 'r') as f:
        args.__dict__ = json.load(f)

    if 'Pretrained' in saved_state:
        args.lr = 0.001
        args.wd = 0.01

    # make dir
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    results_dir = os.path.join("/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo", "Inference results")

    this_result_name = f"{name}_{args.optimizer}_epochs{args.n_epochs}_lr{args.lr}_wd{args.wd}"
    result_path = os.path.join(results_dir, this_result_name)
    os.makedirs(result_path, exist_ok=True)

    # select data

    test_data = pd.read_csv(holdout_set)

    test_trsf = slc.Stream([
            slt.Pad(pad_to=(50,50), padding='r'),
            slt.Crop(crop_mode='c', crop_to=(50, 50)),
    ])

    test_dataset = ImageDataset(dataset=test_data, transforms=test_trsf)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_threads, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)))

    # prep model

    net = utilities.init_model(args, classes=2, TSNE=False)
    net_state_dict = torch.load(saved_state, map_location='cpu', weights_only=True)
    net.load_state_dict(net_state_dict['model'], strict=False)

    # run inference

    net.eval()
    n_batches = len(test_loader)
    device = next(net.parameters()).device
    predictions_list = []
    ground_truth_list = []
    correct = 0
    all_samples = 0
    running_f1 = 0.0
    wrong_prediction_image_paths = []

    progress = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            path = batch['image_path']

            outputs = net(images.float())

            # labels.to('cuda0')
            _ , predictions = torch.max(outputs.data, 1)

            predictions_list.extend(predictions.cpu())
            ground_truth_list.extend(labels.to('cpu').numpy().tolist())

            f1 = f1_score(np.array(ground_truth_list), np.array(predictions_list), average='binary',
                          zero_division='warn')
            running_f1 += f1

            correct += (predictions == labels).sum().item()
            wrong = [k for k in range(len(predictions)) if predictions[k] != labels[k]]
            wrong_prediction_paths = [path[j] for j in wrong]
            wrong_prediction_image_paths.extend(wrong_prediction_paths)

            all_samples = len(np.array(ground_truth_list))

            progress.set_description(
                f"Running inference [{all_samples} | {len(test_data)}] Testing F1-score: {100. * (running_f1 /(i+1)):.3f}, Accuracy: { 100. * (correct / all_samples):.3f}"
            )
            progress.update()
            gc.collect()
            torch.cuda.empty_cache()

        progress.close()

    model_output = {
        "Image path": test_data.Image_path,
        "Prediction": predictions_list,
        "Ground truth": ground_truth_list
    }
    output_df = pd.DataFrame(model_output)
    output_df.to_csv(os.path.join(result_path, "Model_output_results.csv"), index=False)

    f1 = running_f1/ (i+1)
    accuracy = 100. * (correct / all_samples)
    confusion_matrix = c_matrix(ground_truth_list, predictions_list, labels=[0, 1])
    TP = confusion_matrix[0, 0]
    TN = confusion_matrix[1, 1]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    print(f"Final result")
    print(f"F1-score: {100.* f1:.3f}, Accuracy: {accuracy:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, TP: {TP}, TN: {TN}, FN: {FN}, FP: {FP}")

    matrix_labels = ["True", "False"]
    headers = [""] + matrix_labels
    table = [[label] + row.tolist() for label, row in zip(matrix_labels, confusion_matrix)]
    print(tabulate(table, headers=headers, tablefmt="grid"))

    #save results
    model_output_metrics = {"F1-Score": [f1],
                            "Accuracy": [accuracy],
                            "TP": [TP],
                            "TN": [TN],
                            "FN": [FN],
                            "FP": [FP]
                            }
    output_metrics_df = pd.DataFrame(model_output_metrics)
    output_metrics_df.to_csv(os.path.join(result_path, "Model_output_metrics.csv"), index=False)
    #
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=matrix_labels)
    cm_disp.plot()
    plot_name = os.path.join(result_path, f"Confusion_matrix")
    plt.savefig(plot_name)

    if show:
        plt.show()

    del net

    # Apply the t-SNE algorithm to model output

    net = utilities.init_model(args, classes=2, TSNE=True) # Load the model again without the classifier
    net_state_dict = torch.load(saved_state, map_location='cpu', weights_only=False)
    net.load_state_dict(net_state_dict['model'], strict=False) # We are excluding the classifier from the original net
    net.train(False)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device).float()

            # plt.annotate(Path(image_path).stem, (br_x, br_y))
            output = net.forward(images).cpu()
            current_outputs = output.cpu().numpy()
            if i == 0:
                features = current_outputs
            else:
                features = np.concatenate((features, current_outputs))

    tsne = TSNE(n_components=2, perplexity=40).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    #
    tx = tsne_tools.scale_to_01_range(tx)
    ty = tsne_tools.scale_to_01_range(ty)
    #
    images = test_data['Image_path'].values
    wrong_prediction_images = np.array(wrong_prediction_image_paths)
    labels = test_data['Class'].values
    tsne_tools.visualize_with_points(tx, ty, labels, wrong_prediction_images, result_path, show=show)

    tsne_tools.visualize_with_images(tx, ty, images, labels, result_path, wrong_prediction_image_paths, show=show, plot_size=1000, max_image_size=100)

    return tx, ty, labels


def tsne_allfolds():

    # fold0 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_07_18_06_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold0_@epoch10.pth"
    # fold1 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_07_18_06_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold1_@epoch14.pth"
    # fold2 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_07_18_06_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold2_@epoch10.pth"
    # fold3 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_07_18_06_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold3_@epoch13.pth"
    # fold4 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_07_18_06_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold4_@epoch12.pth"

    fold0 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_14_14_55_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Untrained_weights_fold0_@epoch0.pth"
    fold1 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_14_14_55_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Untrained_weights_fold1_@epoch6.pth"
    fold2 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_14_14_55_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Untrained_weights_fold2_@epoch16.pth"
    fold3 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_14_14_55_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Untrained_weights_fold3_@epoch2.pth"
    fold4 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_14_14_55_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Untrained_weights_fold4_@epoch2.pth"


    # Imagenetin painoilla

    # fold0 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_30_15_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold0_@epoch19.pth"
    # fold1 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_30_15_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold1_@epoch3.pth"
    # fold2 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_30_15_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold2_@epoch13.pth"
    # fold3 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_30_15_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold3_@epoch18.pth"
    # fold4 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_30_15_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold4_@epoch9.pth"

    # Koko datasetin painoilla

    # fold0 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_31_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold0_@epoch18.pth"
    # fold1 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_31_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold1_@epoch3.pth"
    # fold2 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_31_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold2_@epoch13.pth"
    # fold3 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_31_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold3_@epoch18.pth"
    # fold4 = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_31_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/VGGNDrop_sgd_best_Pretrained_weights_fold4_@epoch20.pth"

    folds = [fold0, fold1, fold2, fold3, fold4]

    #imagenet testirunin holdout
    # holdout_data_path = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_30_15_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/Data/histo_holdout_metadata.csv"
    # args_path = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_30_15_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/Data/training_arguments.txt"

    # # koko dataset mean std testirunin holdout
    # holdout_data_path = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_31_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/Data/histo_holdout_metadata.csv"
    # args_path = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_17_31_14_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/Data/training_arguments.txt"
    #
    holdout_data_path = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_14_14_55_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/Data/histo_holdout_metadata.csv"
    args_path = "/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Logs/2025_05_12_14_14_55_sgd_epochs20_lr0.01_lr_drop[1, 2, 4, 5]/Data/training_arguments.txt"

    data = []

    for i, fold in enumerate(folds):

        tx, ty, labels = run_inference(args_path, holdout_data_path, fold, name=f"ImageNet_Untrained_fold{i}", show=False)
        data.append([tx, ty, labels])

    df = pd.DataFrame(data)


