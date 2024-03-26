import torch
import h5py
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms 
import os
import os.path as osp

from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import cv2

PIXEL_SCALE = 90.0
THRESHOLDS = [12, 18, 24, 32]
COLOR_MAP = ['lavender', 'indigo', 'mediumblue', 'dodgerblue', 'skyblue', 'cyan',
                                  'olivedrab', 'lime', 'greenyellow', 'orange', 'red', 'magenta', 'pink',]

BOUNDS = [0, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, PIXEL_SCALE]


# COLOR_MAP = np.array([
#     [0, 0, 0,0],
#     [0, 236, 236, 255],
#     [1, 160, 246, 255],
#     [1, 0, 246, 255],
#     [0, 239, 0, 255],
#     [0, 200, 0, 255],
#     [0, 144, 0, 255],
#     [255, 255, 0, 255],
#     [231, 192, 0, 255],
#     [255, 144, 2, 255],
#     [255, 0, 0, 255],
#     [166, 0, 0, 255],
#     [101, 0, 0, 255],
#     [255, 0, 255, 255],
#     [153, 85, 201, 255],
#     [255, 255, 255, 255]
#     ]) / 255

# BOUNDS = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75, 80]
HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255



class Meteo(Dataset):
    def __init__(self, data_path, img_size, type='train', trans=None, seq_len=-1):
        super().__init__()
        
        self.pixel_scale = PIXEL_SCALE
        
        self.data_path = data_path
        self.img_size = img_size

        assert type in ['train', 'test', 'val']
        self.type = type if type!='val' else 'test'
        with h5py.File(data_path,'r') as f:
            self.all_len = int(f[f'{self.type}_len'][()]) #  10000-3000 for train, 2000 for test, 1000 for val
        if trans is not None:
            self.transform = trans
        else:
            self.transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        # transforms.ToTensor(),
                        # trans.Lambda(lambda x: x/255.0),
                        # transforms.Normalize(mean=[0.5], std=[0.5]),
                        # trans.RandomCrop(data_config["img_size"]),

                    ])
                    
    def __len__(self):
        return self.all_len

    def sample(self):
        index = np.random.randint(0, self.all_len)
        return self.__getitem__(index)
    
    
    def __getitem__(self, index):

        with h5py.File(self.data_path,'r') as f:
            imgs = f[self.type][str(index)][()]   # numpy array: (25, 565, 784), dtype=uint8, range(0,70)

            frames = torch.from_numpy(imgs).float().squeeze() 
            frames = frames / self.pixel_scale
            frames = self.transform(frames)     
        return frames.unsqueeze(1) # (25,1,128,128)
 
    
# def gray2color(img):
#     cmap = colors.ListedColormap(COLOR_MAP)
#     norm = colors.BoundaryNorm(BOUNDS, cmap.N)
#     return cmap(norm(img))

def gray2color(image, **kwargs):

    # 定义颜色映射和边界
    cmap = colors.ListedColormap(COLOR_MAP )
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))

    return colored_image



if __name__ == '__main__':
    dataset = Meteo('path/to/dataset/meteo_radar.h5', 128)
    sample1 = dataset.sample()
    sample2 = dataset.sample()
    
    print(len(dataset))