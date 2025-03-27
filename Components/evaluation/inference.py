import os
import gc
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from argparse import ArgumentParser
from faker import Faker
from Components.evaluation import tsne_tools
from Components.training import utilities
from tabulate import tabulate
from Components.data_processing.dataset_splits import ImageDataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix as c_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from tqdm import tqdm

def run_inference(args_path, holdout_set, saved_state, show=False):
    """
    Run inference for a model trained for a binary classification problem
    Creates TSNE visualization for model output
    """
    #Load args
    parser = ArgumentParser()
    args = parser.parse_args()
    with open(args_path, 'r') as f:
        args.__dict__ = json.load(f)

    # make dir
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    results_dir = os.path.join("/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo", "Inference results")
    faker = Faker()
    name = faker.first_name()
    this_result_name = f"{name}_{args.optimizer}_epochs{args.n_epochs}_lr{args.lr}_lr_drop{str(args.lr_drop).replace('[', '').replace(']', '').replace(', ', '-')}_wd{args.wd}"
    result_path = os.path.join(results_dir, this_result_name)
    os.makedirs(result_path, exist_ok=True)

    # select data

    test_data = pd.read_csv(holdout_set)
    test_dataset = ImageDataset(dataset=test_data, transforms=None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_threads, worker_init_fn=lambda wid: np.random.seed(np.uint32(torch.initial_seed() + wid)), pin_memory=True)

    # prep model

    net = utilities.init_model(args, classes=1, TSNE=False)
    net_state_dict = torch.load(saved_state, map_location='cpu', weights_only=False)
    net.load_state_dict(net_state_dict, strict=False)

    # run inference

    net.eval()
    n_batches = test_loader.batch_size
    device = next(net.parameters()).device
    predictions = []
    ground_truth = []
    correct = 0
    all_samples = 0
    running_f1 = 0.0

    progress = tqdm(total=len(test_loader))
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device).float()
            labels = batch['label'].to(device)
            path = batch['image_path']

            outputs = net(images)

            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs > 0.5).int().to('cpu').numpy()

            predictions.extend(preds.tolist())
            ground_truth.extend(labels.to('cpu').numpy().tolist())
            f1 = f1_score(np.array(ground_truth), np.array(predictions), average='binary')
            running_f1 += f1

            correct = np.equal(predictions, ground_truth).sum()
            all_samples = len(ground_truth)

            progress.set_description(
                f"Running inference [{all_samples} | {len(test_data)}] Testing F1-score: {100. * (running_f1 / i):.3f}, Accuracy: { 100. * (correct / all_samples):.3f}"
            )
            progress.update()
            gc.collect()
            torch.cuda.empty_cache()
        progress.close()
    model_output = {
        "Image path": test_data.Image_path,
        "Prediction": predictions,
        "Ground truth": ground_truth
    }
    output_df = pd.DataFrame(model_output)
    output_df.to_csv(os.path.join(result_path, "Model_output_results.csv"), index=False)

    f1 = running_f1/ i
    accuracy = 100. * (correct / all_samples)
    confusion_matrix = c_matrix(ground_truth, predictions, labels=None)
    TP = confusion_matrix[0, 0]
    TN = confusion_matrix[1, 1]
    FN = confusion_matrix[0, 1]
    FP = confusion_matrix[1, 0]
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    print(f"Final result")
    print(f"F1-score: {f1:.3f}, Accuracy: {accuracy:.3f}, Recall: {recall}, Precision: {precision}, TP: {TP}, TN: {TN}, FN: {FN}, FP: {FP}")

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
    if show:
        plt.show()
    plot_name = os.path.join(result_path, f"Confusion_matrix")
    plt.savefig(plot_name)

    # Antin ehottama visualisaatio
    del net
    net = utilities.init_model(args, classes=1, TSNE=True) # Load the model again without the classifier
    net_state_dict = torch.load(saved_state, map_location='cpu', weights_only=False)
    net.load_state_dict(net_state_dict, strict=False) # We are excluding the classifier from the original net
    net.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch['image'].to(device).float()

            output = net.forward(images).cpu()
            current_outputs = output.cpu().numpy()
            if i == 0:
                features = current_outputs
            else:
                features = np.concatenate((features, current_outputs))

    tsne = TSNE(n_components=2, perplexity=10).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]
    #
    tx = tsne_tools.scale_to_01_range(tx)
    ty = tsne_tools.scale_to_01_range(ty)
    #
    images = test_data['Image_path'].values
    labels = test_data['Class'].values
    tsne_tools.visualize_with_points(tx, ty, labels, result_path, show=False)

    tsne_tools.visualize_with_images(tx, ty, images, labels, result_path, show=False, plot_size=1000, max_image_size=100)