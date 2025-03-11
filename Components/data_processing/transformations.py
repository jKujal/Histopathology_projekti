import pandas as pd
from torchvision import transforms
import solt

def image_transformation():

    transform = transforms.Compose([
        #transforms.Pad(padding=1),  #Pad image from 50x50 to 52x52
        transforms.RandomHorizontalFlip(p=0.5),  #Randomly flip some images horizontally
        transforms.CenterCrop(size=(50, 50)),  #Crop back to 50x50 pixels, atm 32x32 before models changes
        transforms.ToTensor(),
    ])
    return transform

def solt_transformations():
    return