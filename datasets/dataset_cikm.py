import numpy as np
import torch
import logging
import h5py

from torch.utils.data import Dataset
from torchvision import transforms
from matplotlib import colors



class CIKM(Dataset):
    """
    cikm.h5:
        sample num:   [train, test, val] = [8000,4000,2000]
        every sample: [T, W, H] = [15, 101, 101]
        data value :  [0,255]
    """
    def __init__(self, data_path, type='train', img_size=128) -> None:
        super().__init__()
        self.data_path = data_path
        assert type in ['train', 'test', 'valid']
        self.type = type
        self.transform = transforms.Compose([
            transforms.CenterCrop((img_size, img_size)),        # padding if img_size > 101
            # transforms.ToTensor(),
            # transforms.Lambda(lambda x: x/40.0-1),
            # transforms.Normalize(mean=[0.5], std=[0.5]),
            # transforms.RandomCrop((img_size, img_size)),

        ])
        with h5py.File(data_path,'r') as f:
            # print(list(f.keys()))
            self.len = f[type+'_len'][()]
            print("{} dataset num:{}".format(type,self.len))
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        key_str = 'sample_'+str(index+1)

        frames = []
        with h5py.File(self.data_path,'r') as f:
            imgs = f[self.type][key_str][()]
            seqs = torch.from_numpy(imgs).type(torch.float32)
            frames = self.transform(seqs)
            frames  = frames / 255.0
            frames = frames.unsqueeze(1)    # [T, 1, W, H]
   
        return frames
    
PIXEL_SCALE = 90.0
# SHIFT = -10.0

COLOR_MAP = np.array([
    [0, 0, 0,0],
    [0, 236, 236, 255],
    [1, 160, 246, 255],
    [1, 0, 246, 255],
    [0, 239, 0, 255],
    [0, 200, 0, 255],
    [0, 144, 0, 255],
    [255, 255, 0, 255],
    [231, 192, 0, 255],
    [255, 144, 2, 255],
    [255, 0, 0, 255],
    [166, 0, 0, 255],
    [101, 0, 0, 255],
    [255, 0, 255, 255],
    [153, 85, 201, 255],
    [255, 255, 255, 255]
    ]) / 255

BOUNDS = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, PIXEL_SCALE]
THRESHOLDS = [20, 30, 35, 40]

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255


def gray2color(image, ***args, **kwargs):

    # 定义颜色映射和边界
    cmap = colors.ListedColormap(COLOR_MAP )
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))

    return colored_image



if __name__ == '__main__':
    dataset = CIKM(data_path='path/to/dataset/cikm.h5',type='valid',img_size=128)
    print(len(dataset))

    sample = dataset.__getitem__(39)

    sample2 = dataset.__getitem__(599)

