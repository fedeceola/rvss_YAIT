import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
from glob import glob
from os import path

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder, crop_ratio, img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.crop_ratio = crop_ratio
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*/*" + self.img_ext))
        self.totensor = transforms.ToTensor()
        # set this in the training loop, at each epoch for each phase
        self.phase = None
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        assert self.phase in ["train", "val"]

        f = self.filenames[idx]        
        img = cv2.imread(f)
        
        if self.crop_ratio > 0:
            idx_crop = int(img.shape[0] * self.crop_ratio)
            img = img[idx_crop:, :, :]

        steering = f.split("/")[-1].split(self.img_ext)[0][6:]
        steering = np.float32(steering)        

        if self.phase == "train":
            if torch.randint(0, 2, (1,)).item():
                img = cv2.flip(img, 1)
                steering *= np.array(-1, dtype=steering.dtype)

        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform[self.phase](img)   
        
        sample = {"image":img , "steering":steering}        
        
        return sample


def test():
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    ds = SteerDataSet("/home/iiticublap205/RVSS_Need4Speed/on_laptop/data",".jpg",transform)

    print("The dataset contains %d images " % len(ds))

    ds_dataloader = DataLoader(ds,batch_size=1,shuffle=True)
    for S in ds_dataloader:
        im = S["image"]    
        y  = S["steering"]
        
        print(im.shape)
        print(y)
        break



if __name__ == "__main__":
    test()
