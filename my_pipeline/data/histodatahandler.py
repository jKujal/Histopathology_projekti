import os
import csv
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

    data_path = Path(r"C:\Users\Joonas\Desktop\copy_histo_images")
    print("Path to the copy of the original images is: " + str(data_path))
    print()

    histo_data_path = Path(r"../../histo_data")

    if os.path.exists(histo_data_path) == True:
        print("Data files already exits")

        return histo_data_path
    else:
        progress = tqdm(total=277524)  # Create progress bar for cool visuals, 277524 = total amount of images
        with open("../../histo_data", 'w', newline='') as file:
            #Creating a .csv-file that contains all of the data
            writer = csv.writer(file)
            writer.writerow(['FolderID', 'Class', 'Image_path'])
            for folderID in os.listdir(data_path):
                #Go through a list of folders, that is created by os.listdir from folders in data_path
                folder_path = os.path.join(data_path, folderID) #Create a path guiding to each folderID
                folder_csv_data = f"{folderID}_data" #I want a csv file for each folderID.

                with open(folder_csv_data, 'w', newline= '') as csvfile:
                    #Need to create a second writer for the folderID csv-files.
                    subject_csv_writer = csv.writer(csvfile)
                    subject_csv_writer.writerow(['FolderID', 'Class', 'Image_path'])
                    for class_number in os.listdir(folder_path):
                        #Class either 0 or 1
                        class_path = os.path.join(folder_path, class_number)

                        for image in os.listdir(class_path):
                            #Go through all of the images, get the full path to it, and use writers to create the full data csv and individual folderID csv
                            #csv-file format: [folderID, class number, full path to image]
                            image_full_path = os.path.join(class_path, image)
                            writer.writerow([folderID, class_number, image_full_path])
                            subject_csv_writer.writerow([folderID, class_number, image_full_path])
                            progress.update()

        return histo_data_path

