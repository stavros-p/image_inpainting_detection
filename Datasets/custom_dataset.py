"""
Custom dataset implementation to make use of did and defacto
dataset

Code Implementation: Stavros Papadopoulos
May 2021
"""

import torch 
from torch.utils.data import Dataset
import numpy as np
import cv2
import os 
from torchvision import transforms
import torch.nn.functional as F



class custom_dataset(Dataset):
    def __init__(self, data_src: str):
        super(custom_dataset ,self).__init__()
        """
        Args:
            data_src(string): Equal to 'did' or 'defacto' depending on the experiment.

        """      
        
        self.images = []
        #self.images_folder = ''
        self.images_path = ''
        self.data_src=data_src
        self.masks=[]
        #self.masks_folder = ''
        self.masks_path = ''

        if data_src == 'did':
            inpaint_methods = ['CA', 'EC', 'GC', 'LB', 'NS', 'PM', 'RN', 'SG', 'SH', 'TE']
            for method in inpaint_methods:
                self.images_path = 'C:/Users/stavr/Desktop/thesis/Datasets/train_dataset/did/DiverseInpaintingDataset/' + method 
                #self.masks_path = self.images_path 
                self.filelist = os.listdir(self.images_path)
                for filename in self.filelist:
                    filepath = os.path.join(self.images_path,filename)
                    if '_mask' in filename:
                        self.masks.append(filepath)
                    else:
                        self.images.append(filepath)

        elif data_src == 'defacto':
            self.images_path = 'C:/Users/stavr/Desktop/thesis/Datasets/train_dataset/defacto/inp_images' 
            self.masks_path = 'C:/Users/stavr/Desktop/thesis/Datasets/train_dataset/defacto/gt_masks'
            self.img_list = sorted(os.listdir(self.images_path))
            self.mask_list = sorted(os.listdir(self.masks_path))

            for filename in self.img_list:
                if '.tif' in filename:
                    self.images.append(filename)
            for filename in self.mask_list:
                if '.tif' in filename:
                    self.masks.append(filename)
        
        elif data_src=='default':
            self.images_path = r'C:\Users\stavr\Desktop\thesis\Datasets\test_dataset\inp_images'
            self.masks_path = r'C:\Users\stavr\Desktop\thesis\Datasets\test_dataset\masks'
            self.img_list = sorted(os.listdir(self.images_path))
            self.mask_list = sorted(os.listdir(self.masks_path))
            for filename in self.img_list:
                self.images.append(filename) 
            for filename in self.mask_list:
                self.masks.append(filename)
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #transforms.CenterCrop(128),
        ])


    def __getitem__(self, idx):
        #returns an image from a particular index 
        return self.load_item(idx)       

    def load_item(self, idx):
        
        #fname1 is the path of the image 
        #fname2 is the path of the gt mask
        if self.data_src== 'defacto': 
            fname1, fname2 = self.images_path +'/' + self.images[idx], self.masks_path +'/' + self.masks[idx]
        elif self.data_src=='did':
            fname1, fname2 = self.images[idx], self.masks[idx]
        elif self.data_src== 'default':
            fname1, fname2 = self.images_path +'/' + self.images[idx], self.masks_path +'/' + self.masks[idx]

        #reads the image
        img = cv2.imread(fname1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(img, (0.5, 0.5), cv2.INTER_LINEAR)
        H, W, _ = img.shape

        #reads the gt mask image
        mask = cv2.imread(fname2)
        #mask = cv2.resize(mask,( 0.5, 0.5), cv2.INTER_LINEAR)

        #converts image and mask to a float array and divides 
        #its value with 255
        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
       
        #returns image, mask and
        #fname1.split('/')[-1] returns the filename
        img=self.transform(img)
        mask=self.tensor(mask[:,:,:1])
        img = F.interpolate(img.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0), scale_factor=0.5, mode='bilinear').squeeze(0)
        return img, mask, fname1.split('/')[-1]

    def __len__(self):
        #retrieves the size of the dataset
        return len(self.masks)


    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)