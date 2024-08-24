
import torch
import os
import numpy as np
import os.path as osp
import datetime

from functools import partial
from matplotlib import colors
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip


HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255


def vis_res(pred_seq, gt_seq, save_path, data_type='vil',
            save_grays=False, do_hmf=False, save_colored=False,save_gif=False,
            pixel_scale = None, thresholds = None, gray2color = None
            ):
    # pred_seq: ndarray, [T, C, H, W], value range: [0, 1] float
    if isinstance(pred_seq, torch.Tensor) or isinstance(gt_seq, torch.Tensor):
        pred_seq = pred_seq.detach().cpu().numpy()
        gt_seq = gt_seq.detach().cpu().numpy()
    pred_seq = pred_seq.squeeze()
    gt_seq = gt_seq.squeeze()
    os.makedirs(save_path, exist_ok=True)

    if save_grays:
        os.makedirs(osp.join(save_path, 'pred'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(pred_seq, gt_seq)):
            
            # cv2.imwrite(osp.join(save_path, 'pred', f'{i}.png'), (pred * PIXEL_SCALE).astype(np.uint8))
            # cv2.imwrite(osp.join(save_path, 'targets', f'{i}.png'), (gt * PIXEL_SCALE).astype(np.uint8))
            
            plt.imsave(osp.join(save_path, 'pred', f'{i}.png'), pred, cmap='gray', vmax=1.0, vmin=0.0)
            plt.imsave(osp.join(save_path, 'targets', f'{i}.png'), gt, cmap='gray', vmax=1.0, vmin=0.0)


    if data_type=='vil':
        pred_seq = pred_seq * pixel_scale
        pred_seq = pred_seq.astype(np.uint8)
        gt_seq = gt_seq * pixel_scale
        gt_seq = gt_seq.astype(np.uint8)
    
    colored_pred = np.array([gray2color(pred_seq[i], data_type=data_type) for i in range(len(pred_seq))], dtype=np.float64)
    colored_gt =  np.array([gray2color(gt_seq[i], data_type=data_type) for i in range(len(gt_seq))],dtype=np.float64)

    if save_colored:
        os.makedirs(osp.join(save_path, 'pred_colored'), exist_ok=True)
        os.makedirs(osp.join(save_path, 'targets_colored'), exist_ok=True)
        for i, (pred, gt) in enumerate(zip(colored_pred, colored_gt)):
            plt.imsave(osp.join(save_path, 'pred_colored', f'{i}.png'), pred)
            plt.imsave(osp.join(save_path, 'targets_colored', f'{i}.png'), gt)


    grid_pred = np.concatenate([
        np.concatenate([i for i in colored_pred], axis=-2),
    ], axis=-3)
    grid_gt = np.concatenate([
        np.concatenate([i for i in colored_gt], axis=-2,),
    ], axis=-3)
    
    grid_concat = np.concatenate([grid_pred, grid_gt], axis=-3,)
    plt.imsave(osp.join(save_path, 'all.png'), grid_concat)
    
    if save_gif:
        clip = ImageSequenceClip(list(colored_pred * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'pred.gif'), fps=4, verbose=False)
        clip = ImageSequenceClip(list(colored_gt * 255), fps=4)
        clip.write_gif(osp.join(save_path, 'targets.gif'), fps=4, verbose=False)
    
    if do_hmf:
        def hit_miss_fa(y_true, y_pred, thres):
            mask = np.zeros_like(y_true)
            mask[np.logical_and(y_true >= thres, y_pred >= thres)] = 4
            mask[np.logical_and(y_true >= thres, y_pred < thres)] = 3
            mask[np.logical_and(y_true < thres, y_pred >= thres)] = 2
            mask[np.logical_and(y_true < thres, y_pred < thres)] = 1
            return mask
            
        grid_pred = np.concatenate([
            np.concatenate([i for i in pred_seq], axis=-1),
        ], axis=-2)
        grid_gt = np.concatenate([
            np.concatenate([i for i in gt_seq], axis=-1),
        ], axis=-2)

        hmf_mask = hit_miss_fa(grid_pred, grid_gt, thres=thresholds[2])
        plt.axis('off')
        plt.imsave(osp.join(save_path, 'hmf.png'), hmf_mask, cmap=colors.ListedColormap(HMF_COLORS))


DATAPATH = {
    'cikm' : 'path/to/dataset/cikm.h5',
    'shanghai' : 'path/to/dataset/shanghai.h5',
    'meteo' : 'path/to/dataset/meteo_radar.h5',
    'sevir' : 'path/to/dataset/sevir'
}

def get_dataset(data_name, img_size, seq_len, **kwargs):
    dataset_name = data_name.lower()
    train = val = test = None

    if dataset_name == 'cikm':
        from .dataset_cikm import CIKM, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train = CIKM(DATAPATH[data_name], 'train', img_size)
        val = CIKM(DATAPATH[data_name], 'valid', img_size)
        test = CIKM(DATAPATH[data_name], 'test', img_size)
        
    elif data_name == 'shanghai':
        from .dataset_shanghai import Shanghai, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Shanghai(DATAPATH[data_name], type='train', img_size=img_size)
        val = Shanghai(DATAPATH[data_name], type='val', img_size=img_size)
        test = Shanghai(DATAPATH[data_name], type='test', img_size=img_size)
    
    elif data_name == 'meteo':
        from .dataset_meteonet import Meteo, gray2color, THRESHOLDS, PIXEL_SCALE
        train = Meteo(DATAPATH[data_name], type='train', img_size=img_size)
        val = Meteo(DATAPATH[data_name], type='val', img_size=img_size)
        test = Meteo(DATAPATH[data_name], type='test', img_size=img_size)
        
    elif dataset_name == 'sevir':
        from .dataset_sevir import SEVIRTorchDataset, gray2color, PIXEL_SCALE, THRESHOLDS
        
        train_valid_split = (2019, 1, 1)
        valid_test_split = (2019, 10, 1)
        batch_size = kwargs.get('batch_size', 1)
        
        train = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            img_size=img_size,
            shuffle=True,
            seq_len=seq_len,
            stride=5,      # ?
            sample_mode='sequent',
            batch_size=batch_size,
            num_shard=1,
            rank=0,
            start_date=None,
            end_date=datetime.datetime(*train_valid_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        val = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            img_size=img_size,
            shuffle=False,
            seq_len=seq_len,
            stride=5,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*train_valid_split),
            end_date=datetime.datetime(*valid_test_split),
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        
        test = SEVIRTorchDataset(
            dataset_dir=DATAPATH[data_name],
            split_mode='uneven',
            shuffle=False,
            img_size=img_size,
            seq_len=seq_len,
            stride=5,      # ?
            sample_mode='sequent',
            batch_size=batch_size * 2,
            num_shard=1,
            rank=0,
            start_date=datetime.datetime(*valid_test_split),
            end_date=None,
            output_type=np.float32,
            preprocess=True,
            rescale_method='01',
            verbose=False
        )
        

    color_fn = partial(vis_res, 
                    pixel_scale = PIXEL_SCALE, 
                    thresholds = THRESHOLDS, 
                    gray2color = gray2color)
    
    return train, val, test, color_fn, PIXEL_SCALE, THRESHOLDS
