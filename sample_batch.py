import os
import os.path as osp
import json
import numpy as np
import cv2

import torch

from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

COLOR_MAP = [[0, 0, 0],
              [0.30196078431372547, 0.30196078431372547, 0.30196078431372547],
              [0.1568627450980392, 0.7450980392156863, 0.1568627450980392],
              [0.09803921568627451, 0.5882352941176471, 0.09803921568627451],
              [0.0392156862745098, 0.4117647058823529, 0.0392156862745098],
              [0.0392156862745098, 0.29411764705882354, 0.0392156862745098],
              [0.9607843137254902, 0.9607843137254902, 0.0],
              [0.9294117647058824, 0.6745098039215687, 0.0],
              [0.9411764705882353, 0.43137254901960786, 0.0],
              [0.6274509803921569, 0.0, 0.0],
              [0.9058823529411765, 0.0, 1.0]]


PIXEL_SCALE = 255.0
BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]


PIXEL_SCALE = 255.0
img_channel = 1
img_size = 128
in_T = 5
out_T = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def gray2color(image):

    cmap = colors.ListedColormap(COLOR_MAP )
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    colored_image = cmap(norm(image))

    return colored_image



def vis_res(pred_seq, gt_seq, save_path, save_grays=False, save_colored=False):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
    pred_seq = pred_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(osp.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):

            plt.imsave(osp.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmax=1.0, vmin=0.0)
            plt.imsave(osp.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmax=1.0, vmin=0.0)

    pred_seq = pred_seq * PIXEL_SCALE
    pred_seq = pred_seq.astype(np.int16)
    gt_seq = gt_seq * PIXEL_SCALE
    gt_seq = gt_seq.astype(np.int16)
    

    colored_pred = np.array([gray2color(pred_seq[i]) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt =  np.array([gray2color(gt_seq[i]) for i in range(len(gt_seq))],dtype=np.float64)
    
    if save_colored:
        os.makedirs(osp.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets_colored'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(osp.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(osp.join(save_path, 'targets_colored', f'{i}.png'), gt)



    clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
    clip.write_gif(osp.join(save_path, 'pred.gif'), fps=4, verbose=False)
    clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
    clip.write_gif(osp.join(save_path, 'targets.gif'), fps=4, verbose=False)



def build_model(chkpt_path):
    from models.phydnet import get_model
    kwargs = {
        "in_shape": (img_channel, img_size, img_size),
        "T_in": in_T,
        "T_out": out_T,
        "device": device
    }
    backbone = get_model(**kwargs)
  
    from diffcast import get_model
    kwargs = {
        'img_channels' : img_channel,
        'dim' : 64,
        'dim_mults' : (1,2,4,8),
        'T_in': in_T,
        'T_out': out_T,
        'sampling_timesteps': 250,
    }
    diff_model = get_model(**kwargs)
    diff_model.load_backbone(backbone)
    
    diff_model = diff_model.to(device)
    
    data = torch.load(chkpt_path, map_location=device)
    print(data.keys())
    diff_model.load_state_dict(data['model'])
    print(f"Loaded model from {chkpt_path}")
    
    return diff_model
    

def read_data(data_path):
    def read_seqs(sample_path):
        targets = []
        for path in sorted(os.listdir(os.path.join(sample_path, "targets")), key=lambda x: int(x.split('.')[0])):
            # print(path)
            if path.endswith(".png"):
                targets.append(cv2.imread(os.path.join(sample_path, 'targets',path), cv2.IMREAD_GRAYSCALE))
        targets = np.array(targets)
        
        inputs = []
        for path in sorted(os.listdir(os.path.join(sample_path, "preds")), key=lambda x: int(x.split('.')[0])):
            # print(path)
            if path.endswith(".png"):
                inputs.append(cv2.imread(os.path.join(sample_path, 'preds', path), cv2.IMREAD_GRAYSCALE))
        inputs = np.array(inputs)
        return targets, inputs

    dirs = sorted(os.listdir(data_path))
    targets, inputs = [], []
    for dir in dirs:
        seq_path = os.path.join(data_path, dir)
        target_seq, input_seq = read_seqs(seq_path)
        targets.append(target_seq)
        inputs.append(input_seq)
    
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs).unsqueeze(2).float() / PIXEL_SCALE
    targets = np.array(targets)
    targets = torch.from_numpy(targets).unsqueeze(2).float() / PIXEL_SCALE
    
    print("Data shape: ", inputs.shape, targets.shape)
    print("Data range: ", inputs.min(), inputs.max(), targets.min(), targets.max())
    
    return inputs.to(device), targets.to(device)



cur_dir = os.path.dirname(os.path.abspath(__file__))
chkpt_path = os.path.join(cur_dir, "resources", "diffcast_phydnet_sevir128.pt")
data_path = os.path.join(cur_dir,"resources", "seqs")

model = build_model(chkpt_path)
input_batch, target_batch = read_data(data_path)

pred_batch, mus, rs = model.sample(input_batch, T_out=out_T)
pred_batch = pred_batch.clamp(0, 1.0)

for b in range(len(pred_batch)):
    save_name = osp.join(cur_dir, "resources", "preds",f"demo_{b}")
    vis_res(pred_batch[b], target_batch[b], save_path=save_name, save_grays=True, save_colored=True)
    vis_res(mus[b], target_batch[b], save_path=save_name+"_mu", save_colored=True)


