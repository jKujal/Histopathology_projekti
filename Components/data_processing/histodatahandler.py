import os
import csv
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold


def create_histo_data():
    """

    :return:
    """

    print("Project path is: " + str(Path.cwd()))
    print("Home path is : " + str(Path.home()))
    print()

    data_path = Path("/home/velkujal/copy_histo_images")
    print("Path to the copy of the original images is: " + str(data_path))
    print()

    histo_data_folder = Path("/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Histo_data")
    os.makedirs(histo_data_folder, exist_ok=True)
    histo_csv_path = os.path.join(histo_data_folder, "All_histo_data")

    if os.path.exists(Path(histo_csv_path)):
        return histo_csv_path
    else:

        with open(histo_csv_path, 'w', newline='') as file:
            progress = tqdm(total=277524)  # 277524 = total amount of images

            # Creating a .csv-file that contains all the data_processing
            writer = csv.writer(file)
            writer.writerow(['FolderID', 'Class', 'Image_path', 'Image_shape', 'x_cor', 'y_cor'])
            for folderID in os.listdir(data_path):
                # Go through a list of folders, that is created by os.listdir from folders in data_path
                folder_path = os.path.join(data_path, folderID)  # Create a path guiding to each folderID

                # folder_csv_data = f"{folderID}_data" #I want a csv file for each folderID.

                # with open(folder_csv_data, 'w', newline= '') as csvfile:
                #     #Need to create a second writer for the folderID csv-files.
                #     subject_csv_writer = csv.writer(csvfile)
                #     subject_csv_writer.writerow(['FolderID', 'Class', 'Image_path'])
                for class_number in os.listdir(folder_path):
                    # Class either 0 or 1
                    class_path = os.path.join(folder_path, class_number)

                    for image in os.listdir(class_path):
                        # Go through all the images, get the full path to it, and use writers to create the full data_processing csv and individual folderID csv
                        # csv-file format: [folderID, class number, full path to image]
                        image_full_path = os.path.join(class_path, image)
                        image = cv2.imread(image_full_path, cv2.IMREAD_UNCHANGED)
                        x_size, y_size, _ = image.shape
                        image_dims = Path(image_full_path).stem.split('_')
                        x_cor = int(image_dims[2].replace('x', ''))
                        y_cor = int(image_dims[3].replace('y', ''))
                        writer.writerow([folderID, class_number, image_full_path, [x_size, y_size], x_cor, y_cor])
                        # subject_csv_writer.writerow([folderID, class_number, image_full_path])
                        progress.update()

    preprocess_data(histo_csv_path)

    return histo_csv_path


def preprocess_data(csv_path):
    data = pd.read_csv(csv_path)
    bad_rows = data[data['Image_shape'] != "[50, 50]"]
    data = data.drop(bad_rows.index, axis=0)

    data = data.to_csv("/home/velkujal/PycharmProjects/UniProject_CV_DL_Histo/Histo_data/All_histo_data", index=False)
    return data
