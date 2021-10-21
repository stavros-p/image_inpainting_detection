import torch 
from torch.utils.data import Dataset
import numpy as np
import cv2

class did_dataset(Dataset):
    def __init__(self, data_root: str, inpaint_method: str):
        super(did_dataset,self).__init__()
        """
        Args:
            data_root (string): Path to the images/masks.
            inpaint_method (string): 1 of the 10 inpainting methods.
        """       
        data_root=data_root
        images=[]
        masks=[]
        test_path=data_root+inpaint_method
        self.filelist = sorted(os.listdir(test_path))
        for filename in self.filelist:
            if '_mask' in filename:
                self.masks.append(filename)
            else:
                self.images.append(filename)
        self.transform = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    def __getitem__(self, idx):
        #returns an image from a particular index 
        return self.load_item(idx)       

    def load_item(self, idx):
        #fname1 is the path of the image 
        #fname2 is the path of the gt mask 
        fname1, fname2 = test_path+'/' + self.images[idx], test_path+'/' + self.masks[idx]
        #reads the image
        img = cv2.imread(fname1)
        H, W, _ = img.shape
        #reads the gt mask image
        mask = cv2.imread(fname2)
        #converts image and mask to a float array and divides 
        #its value with 255
        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        #returns image, mask and
        img=self.transform(img)
        mask=self.tensor(mask[:,:,:1])
        return img, mask, fname1.split('/')[-1]
       # return self.transform(img), self.tensor(mask[:, :, :1]), fname1.split('/')[-1]

    def __len__(self):
        #retrieves the size of the dataset
        return len(images)
    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


