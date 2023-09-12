import pandas as pd
from scipy import ndimage
import torch
from PIL import Image
from torch.utils import data
import cv2
import numpy as np
import torchvision.transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def getData(mode, model_name):
    if mode == 'train':
        df = pd.read_csv('train.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    elif mode == "valid":
        df = pd.read_csv('valid.csv')
        path = df['Path'].tolist()
        label = df['label'].tolist()
        return path, label
    
    else:
        df = pd.read_csv(f'{model_name}_test.csv')
        path = df['Path'].tolist()
        return path, None

  
def mean_filter_denoise(img):
    img_denoised = cv2.boxFilter(img, -1, (1,1), normalize=True)  
    return img_denoised


# 白平衡
'''def simple_white_balance(img):
    avg_b = np.average(img[:, :, 0])
    avg_g = np.average(img[:, :, 1])
    avg_r = np.average(img[:, :, 2])
    avg = (avg_b + avg_g + avg_r) / 3
    scale_b = avg / avg_b
    scale_g = avg / avg_g
    scale_r = avg / avg_r
    img[:, :, 0] = cv2.multiply(img[:, :, 0], scale_b)
    img[:, :, 1] = cv2.multiply(img[:, :, 1], scale_g)
    img[:, :, 2] = cv2.multiply(img[:, :, 2], scale_r)
    return img'''

# 影像增強
def clahe_enhancement(img):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(9,9))
    channels = cv2.split(img)
    channels = list(channels)  # Convert tuple to list
    for i in range(len(channels)):
        channels[i] = clahe.apply(channels[i])
    img_enhanced = cv2.merge(channels)
    return img_enhanced

# Gabor Filter
'''def build_gabor_filter(size, sigma, theta, lambd, gamma, psi):
    return cv2.getGaborKernel((size, size), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)

class GaborFilter(object):
    def __init__(self, size, sigma, theta, lambd, gamma, psi):
        self.size = size
        self.sigma = sigma
        self.theta = theta
        self.lambd = lambd
        self.gamma = gamma
        self.psi = psi

    def __call__(self, img):
        gabor_filter = build_gabor_filter(self.size, self.sigma, self.theta, self.lambd, self.gamma, self.psi)
        return Image.fromarray(cv2.filter2D(np.array(img), -1, gabor_filter))'''


# Unsharp Masking
def unsharp_mask(image, sigma, strength):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened


# High-pass Filter
'''def high_pass_filter(image, size):
    image = Image.fromarray(image.astype(np.uint8)) if isinstance(image, np.ndarray) else image
    low_pass = ndimage.gaussian_filter(image, size)
    high_pass = image - low_pass
    return high_pass

class HighPassFilter(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return Image.fromarray(high_pass_filter(np.array(img), self.size))'''


class LeukemiaLoader(data.Dataset):
    def __init__(self, mode, model_name):

        self.img_name, self.label = getData(mode, model_name)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  
        
        
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),  
                transforms.CenterCrop(270),
                transforms.Resize((224, 224)),
                #GaborFilter(size=5, sigma=1.0, theta=0, lambd=1.0, gamma=0.5, psi=0),
                #HighPassFilter(size=3),
                transforms.ColorJitter(brightness=0.2, contrast=1.2, saturation=1.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0026741,  0.00156906, 0.00272411], std=[0.00314106, 0.00195709, 0.00304743])
            ])
        else :
            self.transform = transforms.Compose([
                transforms.CenterCrop(270),
                transforms.Resize((224, 224)),
                #GaborFilter(size=5, sigma=1.0, theta=0, lambd=1.0, gamma=0.5, psi=0),
                #HighPassFilter(size=3),
                #transforms.ColorJitter(brightness=0.2, contrast=1.2, saturation=1.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0026741,  0.00156906, 0.00272411], std=[0.00314106, 0.00195709, 0.00304743])
            ])
        

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):

        # 獲取圖片路徑並加載圖片
        path = self.img_name[index]



        img = cv2.imread(path, cv2.IMREAD_COLOR)
        
        #img = mean_filter_denoise(img)
        img = clahe_enhancement(img)
        img = unsharp_mask(img, 3, 1.5)
        
        
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        #img = simple_white_balance(img)
        img = Image.fromarray(img)
        img = self.transform(img)
        


        # 獲取真實標籤
        if self.mode == 'train' or self.mode == 'valid':
            label = self.label[index]
            return img, label
        else:
            return img
