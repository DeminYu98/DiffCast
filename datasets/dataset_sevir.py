"""Code is adapted from https://github.com/MIT-AI-Accelerator/neurips-2020-sevir. Their license is MIT License."""

import os
import os.path as osp
from typing import List, Union, Dict, Sequence
from math import ceil
import numpy as np
import numpy.random as nprand
import datetime
import pandas as pd
import h5py 
import cv2

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn.functional import avg_pool2d
from torchvision import transforms 

from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

def change_layout_np(data,
                     in_layout='NHWT', out_layout='NHWT',
                     ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 2, 3, 1))
    elif in_layout == 'NTWHC':
        data = data[:, :, :, :, 0]
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
    elif in_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = np.transpose(data,
                            axes=(1, 2, 3, 0))
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
    elif out_layout == 'NWHT':
        data = np.transpose(data,
                            axes=(0, 2, 1, 3))
    elif out_layout == 'NTCHW':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=2)
    elif out_layout == 'NTHWC':
        data = np.transpose(data,
                            axes=(0, 3, 1, 2))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'NTWHC':
        data = np.transpose(data,
                            axes=(0, 3, 2, 1))
        data = np.expand_dims(data, axis=-1)
    elif out_layout == 'TNHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
    elif out_layout == 'TNCHW':
        data = np.transpose(data,
                            axes=(3, 0, 1, 2))
        data = np.expand_dims(data, axis=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.ascontiguousarray()
    return data

def change_layout_torch(data,
                        in_layout='NHWT', out_layout='NHWT',
                        ret_contiguous=False):
    # first convert to 'NHWT'
    if in_layout == 'NHWT':
        pass
    elif in_layout == 'NTHW':
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'NTHWC':
        data = data[:, :, :, :, 0]
        data = data.permute(0, 2, 3, 1)
    elif in_layout == 'TNHW':
        data = data.permute(1, 2, 3, 0)
    elif in_layout == 'TNCHW':
        data = data[:, :, 0, :, :]
        data = data.permute(1, 2, 3, 0)
    else:
        raise NotImplementedError

    if out_layout == 'NHWT':
        pass
    elif out_layout == 'NTHW':
        data = data.permute(0, 3, 1, 2)
    elif out_layout == 'NTCHW':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    elif out_layout == 'NTHWC':
        data = data.permute(0, 3, 1, 2)
        data = torch.unsqueeze(data, dim=-1)
    elif out_layout == 'TNHW':
        data = data.permute(3, 0, 1, 2)
    elif out_layout == 'TNCHW':
        data = data.permute(3, 0, 1, 2)
        data = torch.unsqueeze(data, dim=2)
    else:
        raise NotImplementedError
    if ret_contiguous:
        data = data.contiguous()
    return data


# SEVIR Dataset constants
SEVIR_DATA_TYPES = ['vis', 'ir069', 'ir107', 'vil', 'lght']
SEVIR_RAW_DTYPES = {'vis': np.int16,
                    'ir069': np.int16,
                    'ir107': np.int16,
                    'vil': np.int16,
                    'lght': np.int16}
LIGHTING_FRAME_TIMES = np.arange(- 120.0, 125.0, 5) * 60
SEVIR_DATA_SHAPE = {'lght': (48, 48), }
PREPROCESS_SCALE_SEVIR = {'vis': 1,  # Not utilized in original paper
                          'ir069': 1 / 1174.68,
                          'ir107': 1 / 2562.43,
                          'vil': 1 / 47.54,
                          'lght': 1 / 0.60517}
PREPROCESS_OFFSET_SEVIR = {'vis': 0,  # Not utilized in original paper
                           'ir069': 3683.58,
                           'ir107': 1552.80,
                           'vil': - 33.44,
                           'lght': - 0.02990}
PREPROCESS_SCALE_01 = {'vis': 1,
                       'ir069': 1,
                       'ir107': 1,
                       'vil': 1 / 255,  # currently the only one implemented
                       'lght': 1}
PREPROCESS_OFFSET_01 = {'vis': 0,
                        'ir069': 0,
                        'ir107': 0,
                        'vil': 0,  # currently the only one implemented
                        'lght': 0}



class SEVIRDataLoader:
    r"""
    DataLoader that loads SEVIR sequences, and spilts each event
    into segments according to specified sequence length.

    Event Frames:
        [-----------------------raw_seq_len----------------------]
        [-----seq_len-----]
        <--stride-->[-----seq_len-----]
                    <--stride-->[-----seq_len-----]
                                        ...
    """
    def __init__(self,
                 dataset_dir: str,
                 data_types: Sequence[str] = None,
                 seq_len: int = 49,
                 raw_seq_len: int = 49,
                 sample_mode: str = 'sequent',
                 stride: int = 12,
                 batch_size: int = 1,
                 layout: str = 'NHWT',
                 num_shard: int = 1,
                 rank: int = 0,
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter=None,
                 catalog_filter='default',
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type=np.float32,
                 preprocess: bool = True,
                 rescale_method: str = '01',
                 downsample_dict: Dict[str, Sequence[int]] = None,
                 verbose: bool = False):
        r"""
        Parameters
        ----------
        data_types
            A subset of SEVIR_DATA_TYPES.
        seq_len
            The length of the data sequences. Should be smaller than the max length raw_seq_len.
        raw_seq_len
            The length of the raw data sequences.
        sample_mode
            'random' or 'sequent'
        stride
            Useful when sample_mode == 'sequent'
            stride must not be smaller than out_len to prevent data leakage in testing.
        batch_size
            Number of sequences in one batch.
        layout
            str: consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
            The layout of sampled data. Raw data layout is 'NHWT'.
            valid layout: 'NHWT', 'NTHW', 'NTCHW', 'TNHW', 'TNCHW'.
        num_shard
            Split the whole dataset into num_shard parts for distributed training.
        rank
            Rank of the current process within num_shard.
        split_mode: str
            if 'ceil', all `num_shard` dataloaders have the same length = ceil(total_len / num_shard).
            Different dataloaders may have some duplicated data batches, if the total size of datasets is not divided by num_shard.
            if 'floor', all `num_shard` dataloaders have the same length = floor(total_len / num_shard).
            The last several data batches may be wasted, if the total size of datasets is not divided by num_shard.
            if 'uneven', the last datasets has larger length when the total length is not divided by num_shard.
            The uneven split leads to synchronization error in dist.all_reduce() or dist.barrier().
            See related issue: https://github.com/pytorch/pytorch/issues/33148
            Notice: this also affects the behavior of `self.use_up`.
        sevir_catalog
            Name of SEVIR catalog CSV file.
        sevir_data_dir
            Directory path to SEVIR data.
        start_date
            Start time of SEVIR samples to generate.
        end_date
            End time of SEVIR samples to generate.
        datetime_filter
            function
            Mask function applied to time_utc column of catalog (return true to keep the row).
            Pass function of the form   lambda t : COND(t)
            Example:  lambda t: np.logical_and(t.dt.hour>=13,t.dt.hour<=21)  # Generate only day-time events
        catalog_filter
            function or None or 'default'
            Mask function applied to entire catalog dataframe (return true to keep row).
            Pass function of the form lambda catalog:  COND(catalog)
            Example:  lambda c:  [s[0]=='S' for s in c.id]   # Generate only the 'S' events
        shuffle
            bool, If True, data samples are shuffled before each epoch.
        shuffle_seed
            int, Seed to use for shuffling.
        output_type
            np.dtype, dtype of generated tensors
        preprocess
            bool, If True, self.preprocess_data_dict(data_dict) is called before each sample generated
        downsample_dict:
            dict, downsample_dict.keys() == data_types. downsample_dict[key] is a Sequence of (t_factor, h_factor, w_factor),
            representing the downsampling factors of all dimensions.
        verbose
            bool, verbose when opening raw data files
        """
        super(SEVIRDataLoader, self).__init__()

        if data_types is None:
            data_types = SEVIR_DATA_TYPES
        else:
            assert set(data_types).issubset(SEVIR_DATA_TYPES)
        
        self.dataset_dir = dataset_dir
        sevir_catalog = os.path.join(dataset_dir, "CATALOG.csv")

        sevir_data_dir = os.path.join(dataset_dir, "data")


        # configs which should not be modified
        self._dtypes = SEVIR_RAW_DTYPES
        self.lght_frame_times = LIGHTING_FRAME_TIMES
        self.data_shape = SEVIR_DATA_SHAPE

        self.raw_seq_len = raw_seq_len
        assert seq_len <= self.raw_seq_len, f'seq_len must not be larger than raw_seq_len = {raw_seq_len}, got {seq_len}.'
        self.seq_len = seq_len
        assert sample_mode in ['random', 'sequent'], f'Invalid sample_mode = {sample_mode}, must be \'random\' or \'sequent\'.'
        self.sample_mode = sample_mode
        self.stride = stride
        self.batch_size = batch_size
        valid_layout = ('NHWT', 'NTHW', 'NTCHW', 'NTHWC', 'TNHW', 'TNCHW')
        if layout not in valid_layout:
            raise ValueError(f'Invalid layout = {layout}! Must be one of {valid_layout}.')
        self.layout = layout
        self.num_shard = num_shard
        self.rank = rank
        valid_split_mode = ('ceil', 'floor', 'uneven')
        if split_mode not in valid_split_mode:
            raise ValueError(f'Invalid split_mode: {split_mode}! Must be one of {valid_split_mode}.')
        self.split_mode = split_mode
        self._samples = None
        self._hdf_files = {}
        self.data_types = data_types
        if isinstance(sevir_catalog, str):
            self.catalog = pd.read_csv(sevir_catalog, parse_dates=['time_utc'], low_memory=False)
        else:
            self.catalog = sevir_catalog
        self.sevir_data_dir = sevir_data_dir
        self.datetime_filter = datetime_filter
        self.catalog_filter = catalog_filter
        self.start_date = start_date
        self.end_date = end_date
        self.shuffle = shuffle
        self.shuffle_seed = int(shuffle_seed)
        self.output_type = output_type
        self.preprocess = preprocess
        self.downsample_dict = downsample_dict
        self.rescale_method = rescale_method
        self.verbose = verbose

        if self.start_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc > self.start_date]
        if self.end_date is not None:
            self.catalog = self.catalog[self.catalog.time_utc <= self.end_date]
        if self.datetime_filter:
            self.catalog = self.catalog[self.datetime_filter(self.catalog.time_utc)]

        if self.catalog_filter is not None:
            if self.catalog_filter == 'default':
                self.catalog_filter = lambda c: c.pct_missing == 0
            self.catalog = self.catalog[self.catalog_filter(self.catalog)]

        self._compute_samples()
        self._open_files(verbose=self.verbose)
        self.reset()

    def _compute_samples(self):
        """
        Computes the list of samples in catalog to be used. This sets self._samples
        """
        # locate all events containing colocated data_types
        imgt = self.data_types
        imgts = set(imgt)
        filtcat = self.catalog[ np.logical_or.reduce([self.catalog.img_type==i for i in imgt]) ]
        # remove rows missing one or more requested img_types
        filtcat = filtcat.groupby('id').filter(lambda x: imgts.issubset(set(x['img_type'])))
        # If there are repeated IDs, remove them (this is a bug in SEVIR)
        # TODO: is it necessary to keep one of them instead of deleting them all
        filtcat = filtcat.groupby('id').filter(lambda x: x.shape[0]==len(imgt))
        self._samples = filtcat.groupby('id').apply(lambda df: self._df_to_series(df,imgt) )
        if self.shuffle:
            self.shuffle_samples()

    def shuffle_samples(self):
        self._samples = self._samples.sample(frac=1, random_state=self.shuffle_seed)

    def _df_to_series(self, df, imgt):
        d = {}
        df = df.set_index('img_type')
        for i in imgt:
            s = df.loc[i]
            idx = s.file_index if i != 'lght' else s.id
            d.update({f'{i}_filename': [s.file_name],
                      f'{i}_index': [idx]})

        return pd.DataFrame(d)

    def _open_files(self, verbose=True):
        """
        Opens HDF files
        """
        imgt = self.data_types
        hdf_filenames = []
        for t in imgt:
            hdf_filenames += list(np.unique( self._samples[f'{t}_filename'].values ))
        self._hdf_files = {}
        for f in hdf_filenames:
            if verbose:
                print('Opening HDF5 file for reading', f)
            self._hdf_files[f] = h5py.File(self.sevir_data_dir + '/' + f, 'r')

    def close(self):
        """
        Closes all open file handles
        """
        for f in self._hdf_files:
            self._hdf_files[f].close()
        self._hdf_files = {}

    @property
    def num_seq_per_event(self):
        return 1 + (self.raw_seq_len - self.seq_len) // self.stride

    @property
    def total_num_seq(self):
        """
        The total number of sequences within each shard.
        Notice that it is not the product of `self.num_seq_per_event` and `self.total_num_event`.
        """
        return int(self.num_seq_per_event * self.num_event)

    @property
    def total_num_event(self):
        """
        The total number of events in the whole dataset, before split into different shards.
        """
        return int(self._samples.shape[0])

    @property
    def start_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx >= start_event_idx
        """
        return self.total_num_event // self.num_shard * self.rank

    @property
    def end_event_idx(self):
        """
        The event idx used in certain rank should satisfy event_idx < end_event_idx

        """
        if self.split_mode == 'ceil':
            _last_start_event_idx = self.total_num_event // self.num_shard * (self.num_shard - 1)
            _num_event = self.total_num_event - _last_start_event_idx
            return self.start_event_idx + _num_event
        elif self.split_mode == 'floor':
            return self.total_num_event // self.num_shard * (self.rank + 1)
        else:  # self.split_mode == 'uneven':
            if self.rank == self.num_shard - 1:  # the last process
                return self.total_num_event
            else:
                return self.total_num_event // self.num_shard * (self.rank + 1)

    @property
    def num_event(self):
        """
        The number of events split into each rank
        """
        return self.end_event_idx - self.start_event_idx

    def _read_data(self, row, data):
        """
        Iteratively read data into data dict. Finally data[imgt] gets shape (batch_size, height, width, raw_seq_len).

        Parameters
        ----------
        row
            A series with fields IMGTYPE_filename, IMGTYPE_index, IMGTYPE_time_index.
        data
            Dict, data[imgt] is a data tensor with shape = (tmp_batch_size, height, width, raw_seq_len).

        Returns
        -------
        data
            Updated data. Updated shape = (tmp_batch_size + 1, height, width, raw_seq_len).
        """
        imgtyps = np.unique([x.split('_')[0] for x in list(row.keys())])
        for t in imgtyps:
            fname = row[f'{t}_filename']
            idx = row[f'{t}_index']
            t_slice = slice(0, None)
            # Need to bin lght counts into grid
            if t == 'lght':
                lght_data = self._hdf_files[fname][idx][:]
                data_i = self._lght_to_grid(lght_data, t_slice)
            else:
                data_i = self._hdf_files[fname][t][idx:idx + 1, :, :, t_slice]
            data[t] = np.concatenate((data[t], data_i), axis=0) if (t in data) else data_i

        return data

    def _lght_to_grid(self, data, t_slice=slice(0, None)):
        """
        Converts Nx5 lightning data matrix into a 2D grid of pixel counts
        """
        # out_size = (48,48,len(self.lght_frame_times)-1) if isinstance(t_slice,(slice,)) else (48,48)
        out_size = (*self.data_shape['lght'], len(self.lght_frame_times)) if t_slice.stop is None else (*self.data_shape['lght'], 1)
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # filter out points outside the grid
        x, y = data[:, 3], data[:, 4]
        m = np.logical_and.reduce([x >= 0, x < out_size[0], y >= 0, y < out_size[1]])
        data = data[m, :]
        if data.shape[0] == 0:
            return np.zeros((1,) + out_size, dtype=np.float32)

        # Filter/separate times
        t = data[:, 0]
        if t_slice.stop is not None:  # select only one time bin
            if t_slice.stop > 0:
                if t_slice.stop < len(self.lght_frame_times):
                    tm = np.logical_and(t >= self.lght_frame_times[t_slice.stop - 1],
                                        t < self.lght_frame_times[t_slice.stop])
                else:
                    tm = t >= self.lght_frame_times[-1]
            else:  # special case:  frame 0 uses lght from frame 1
                tm = np.logical_and(t >= self.lght_frame_times[0], t < self.lght_frame_times[1])
            # tm=np.logical_and( (t>=FRAME_TIMES[t_slice],t<FRAME_TIMES[t_slice+1]) )

            data = data[tm, :]
            z = np.zeros(data.shape[0], dtype=np.int64)
        else:  # compute z coordinate based on bin location times
            z = np.digitize(t, self.lght_frame_times) - 1
            z[z == -1] = 0  # special case:  frame 0 uses lght from frame 1

        x = data[:, 3].astype(np.int64)
        y = data[:, 4].astype(np.int64)

        k = np.ravel_multi_index(np.array([y, x, z]), out_size)
        n = np.bincount(k, minlength=np.prod(out_size))
        return np.reshape(n, out_size).astype(np.int16)[np.newaxis, :]


    @property
    def sample_count(self):
        """
        Record how many times self.__next__() is called.
        """
        return self._sample_count

    def inc_sample_count(self):
        self._sample_count += 1

    @property
    def curr_event_idx(self):
        return self._curr_event_idx

    @property
    def curr_seq_idx(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self._curr_seq_idx

    def set_curr_event_idx(self, val):
        self._curr_event_idx = val

    def set_curr_seq_idx(self, val):
        """
        Used only when self.sample_mode == 'sequent'
        """
        self._curr_seq_idx = val

    def reset(self, shuffle: bool = None):
        self.set_curr_event_idx(val=self.start_event_idx)
        self.set_curr_seq_idx(0)
        self._sample_count = 0
        if shuffle is None:
            shuffle = self.shuffle
        if shuffle:
            self.shuffle_samples()

    def __len__(self):
        """
        Used only when self.sample_mode == 'sequent'
        """
        return self.total_num_seq // self.batch_size

    @property
    def use_up(self):
        """
        Check if dataset is used up in 'sequent' mode.
        """
        if self.sample_mode == 'random':
            return False
        else:   # self.sample_mode == 'sequent'
            # compute the remaining number of sequences in current event
            curr_event_remain_seq = self.num_seq_per_event - self.curr_seq_idx
            all_remain_seq = curr_event_remain_seq + (
                        self.end_event_idx - self.curr_event_idx - 1) * self.num_seq_per_event
            if self.split_mode == "floor":
                # This approach does not cover all available data, but avoid dealing with masks
                return all_remain_seq < self.batch_size
            else:
                return all_remain_seq <= 0

    def _load_event_batch(self, event_idx, event_batch_size):
        """
        Loads a selected batch of events (not batch of sequences) into memory.

        Parameters
        ----------
        idx
        event_batch_size
            event_batch[i] = all_type_i_available_events[idx:idx + event_batch_size]
        Returns
        -------
        event_batch
            list of event batches.
            event_batch[i] is the event batch of the i-th data type.
            Each event_batch[i] is a np.ndarray with shape = (event_batch_size, height, width, raw_seq_len)
        """
        event_idx_slice_end = event_idx + event_batch_size
        pad_size = 0
        if event_idx_slice_end > self.end_event_idx:
            pad_size = event_idx_slice_end - self.end_event_idx
            event_idx_slice_end = self.end_event_idx
        pd_batch = self._samples.iloc[event_idx:event_idx_slice_end]
        data = {}
        for index, row in pd_batch.iterrows():
            data = self._read_data(row, data)
        if pad_size > 0:
            event_batch = []
            for t in self.data_types:
                pad_shape = [pad_size, ] + list(data[t].shape[1:])
                data_pad = np.concatenate((data[t].astype(self.output_type),
                                           np.zeros(pad_shape, dtype=self.output_type)),
                                          axis=0)
                event_batch.append(data_pad)
        else:
            event_batch = [data[t].astype(self.output_type) for t in self.data_types]
        return event_batch

    def __iter__(self):
        return self

    def __next__(self):
        if self.sample_mode == 'random':
            self.inc_sample_count()
            ret_dict = self._random_sample()
        else:
            if self.use_up:
                raise StopIteration
            else:
                self.inc_sample_count()
                ret_dict = self._sequent_sample()
        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict,
                                            data_types=self.data_types)
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 layout=self.layout,
                                                 rescale=self.rescale_method)
        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 factors_dict=self.downsample_dict,
                                                 layout=self.layout)
        return ret_dict

    def __getitem__(self, index):
        data_dict = self._idx_sample(index=index)
        return data_dict

    @staticmethod
    def preprocess_data_dict(data_dict, data_types=None, layout='NHWT', rescale='01'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
        data_types: Sequence[str]
            The data types that we want to rescale. This mainly excludes "mask" from preprocessing.
        layout: str
            consists of batch_size 'N', seq_len 'T', channel 'C', height 'H', width 'W'
        rescale:    str
            'sevir': use the offsets and scale factors in original implementation.
            '01': scale all values to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict:  Dict[str, Union[np.ndarray, torch.Tensor]]
            preprocessed data
        """
        if rescale == 'sevir':
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == '01':
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, np.ndarray):
                    data = scale_dict[key] * (
                            data.astype(np.float32) +
                            offset_dict[key])
                    data = change_layout_np(data=data,
                                            in_layout='NHWT',
                                            out_layout=layout)
                elif isinstance(data, torch.Tensor):
                    data = scale_dict[key] * (
                            data.float() +
                            offset_dict[key])
                    data = change_layout_torch(data=data,
                                               in_layout='NHWT',
                                               out_layout=layout)
                data_dict[key] = data
        return data_dict

    @staticmethod
    def process_data_dict_back(data_dict, data_types=None, rescale='01'):
        """
        Parameters
        ----------
        data_dict
            each data_dict[key] is a torch.Tensor.
        rescale
            str:
                'sevir': data are scaled using the offsets and scale factors in original implementation.
                '01': data are all scaled to range 0 to 1, currently only supports 'vil'
        Returns
        -------
        data_dict
            each data_dict[key] is the data processed back in torch.Tensor.
        """
        if rescale == 'sevir':
            scale_dict = PREPROCESS_SCALE_SEVIR
            offset_dict = PREPROCESS_OFFSET_SEVIR
        elif rescale == '01':
            scale_dict = PREPROCESS_SCALE_01
            offset_dict = PREPROCESS_OFFSET_01
        else:
            raise ValueError(f'Invalid rescale option: {rescale}.')
        if data_types is None:
            data_types = data_dict.keys()
        for key in data_types:
            data = data_dict[key]
            data = data.float() / scale_dict[key] - offset_dict[key]
            data_dict[key] = data
        return data_dict

    @staticmethod
    def data_dict_to_tensor(data_dict, data_types=None):
        """
        Convert each element in data_dict to torch.Tensor (copy without grad).
        """
        ret_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        for key, data in data_dict.items():
            if key in data_types:
                if isinstance(data, torch.Tensor):
                    ret_dict[key] = data.detach().clone()
                elif isinstance(data, np.ndarray):
                    ret_dict[key] = torch.from_numpy(data)
                else:
                    raise ValueError(f"Invalid data type: {type(data)}. Should be torch.Tensor or np.ndarray")
            else:   # key == "mask"
                ret_dict[key] = data
        return ret_dict

    @staticmethod
    def downsample_data_dict(data_dict, data_types=None, factors_dict=None, layout='NHWT'):
        """
        Parameters
        ----------
        data_dict:  Dict[str, Union[np.array, torch.Tensor]]
        factors_dict:   Optional[Dict[str, Sequence[int]]]
            each element `factors` is a Sequence of int, representing (t_factor, h_factor, w_factor)

        Returns
        -------
        downsampled_data_dict:  Dict[str, torch.Tensor]
            Modify on a deep copy of data_dict instead of directly modifying the original data_dict
        """
        if factors_dict is None:
            factors_dict = {}
        if data_types is None:
            data_types = data_dict.keys()
        downsampled_data_dict = SEVIRDataLoader.data_dict_to_tensor(
            data_dict=data_dict,
            data_types=data_types)    # make a copy
        for key, data in data_dict.items():
            factors = factors_dict.get(key, None)
            if factors is not None:
                downsampled_data_dict[key] = change_layout_torch(
                    data=downsampled_data_dict[key],
                    in_layout=layout,
                    out_layout='NTHW')
                # downsample t dimension
                t_slice = [slice(None, None), ] * 4
                t_slice[1] = slice(None, None, factors[0])
                downsampled_data_dict[key] = downsampled_data_dict[key][tuple(t_slice)]
                # downsample spatial dimensions
                downsampled_data_dict[key] = avg_pool2d(
                    input=downsampled_data_dict[key],
                    kernel_size=(factors[1], factors[2]))

                downsampled_data_dict[key] = change_layout_torch(
                    data=downsampled_data_dict[key],
                    in_layout='NTHW',
                    out_layout=layout)

        return downsampled_data_dict

    def _random_sample(self):
        """
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        num_sampled = 0
        event_idx_list = nprand.randint(low=self.start_event_idx,
                                        high=self.end_event_idx,
                                        size=self.batch_size)
        seq_idx_list = nprand.randint(low=0,
                                      high=self.num_seq_per_event,
                                      size=self.batch_size)
        seq_slice_list = [slice(seq_idx * self.stride,
                                seq_idx * self.stride + self.seq_len)
                          for seq_idx in seq_idx_list]
        ret_dict = {}
        while num_sampled < self.batch_size:
            event = self._load_event_batch(event_idx=event_idx_list[num_sampled],
                                           event_batch_size=1)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event[imgt_idx][[0, ], :, :, seq_slice_list[num_sampled]]  # keep the dim of batch_size for concatenation
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
        return ret_dict

    def _sequent_sample(self):
        """
        Returns
        -------
        ret_dict:   Dict
            `ret_dict.keys()` contains `self.data_types`.
            `ret_dict["mask"]` is a list of bool, indicating if the data entry is real or padded.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        assert not self.use_up, 'Data loader used up! Reset it to reuse.'
        event_idx = self.curr_event_idx
        seq_idx = self.curr_seq_idx
        num_sampled = 0
        sampled_idx_list = []   # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx,
                                     'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx,
                                             event_batch_size=event_batch_size)
        ret_dict = {"mask": []}
        all_no_pad_flag = True
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx, ]  # use [] to keepdim
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                              sampled_idx['seq_idx'] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})
            # add mask
            no_pad_flag = sampled_idx['event_idx'] < self.end_event_idx
            if not no_pad_flag:
                all_no_pad_flag = False
            ret_dict["mask"].append(no_pad_flag)
        if all_no_pad_flag:
            # if there is no padded data items at all, set `ret_dict["mask"] = None` for convenience.
            ret_dict["mask"] = None
        # update current idx
        self.set_curr_event_idx(event_idx)
        self.set_curr_seq_idx(seq_idx)
        return ret_dict

    def _idx_sample(self, index):
        """
        Parameters
        ----------
        index
            The index of the batch to sample.
        Returns
        -------
        ret_dict
            dict. ret_dict.keys() == self.data_types.
            If self.preprocess == False:
                ret_dict[imgt].shape == (batch_size, height, width, seq_len)
        """
        event_idx = (index * self.batch_size) // self.num_seq_per_event
        seq_idx = (index * self.batch_size) % self.num_seq_per_event
        num_sampled = 0
        sampled_idx_list = []  # list of (event_idx, seq_idx) records
        while num_sampled < self.batch_size:
            sampled_idx_list.append({'event_idx': event_idx,
                                     'seq_idx': seq_idx})
            seq_idx += 1
            if seq_idx >= self.num_seq_per_event:
                event_idx += 1
                seq_idx = 0
            num_sampled += 1

        start_event_idx = sampled_idx_list[0]['event_idx']
        event_batch_size = sampled_idx_list[-1]['event_idx'] - start_event_idx + 1

        event_batch = self._load_event_batch(event_idx=start_event_idx,
                                             event_batch_size=event_batch_size)
        ret_dict = {}
        for sampled_idx in sampled_idx_list:
            batch_slice = [sampled_idx['event_idx'] - start_event_idx, ]  # use [] to keepdim
            seq_slice = slice(sampled_idx['seq_idx'] * self.stride,
                              sampled_idx['seq_idx'] * self.stride + self.seq_len)
            for imgt_idx, imgt in enumerate(self.data_types):
                sampled_seq = event_batch[imgt_idx][batch_slice, :, :, seq_slice]
                if imgt in ret_dict:
                    ret_dict[imgt] = np.concatenate((ret_dict[imgt], sampled_seq),
                                                    axis=0)
                else:
                    ret_dict.update({imgt: sampled_seq})

        ret_dict = self.data_dict_to_tensor(data_dict=ret_dict,
                                            data_types=self.data_types)
        if self.preprocess:
            ret_dict = self.preprocess_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 layout=self.layout,
                                                 rescale=self.rescale_method)

        if self.downsample_dict is not None:
            ret_dict = self.downsample_data_dict(data_dict=ret_dict,
                                                 data_types=self.data_types,
                                                 factors_dict=self.downsample_dict,
                                                 layout=self.layout)
        return ret_dict

class SEVIRTorchDataset(TorchDataset):

    def __init__(self,
                 dataset_dir: str,
                 seq_len: int = 25,
                 img_size: int = 128,
                 raw_seq_len: int = 49,
                 sample_mode: str = "sequent",
                 stride: int = 20,
                 batch_size: int = 1,
                 layout: str = "NTHW",
                 num_shard: int = 1,
                 rank: int = 0,
                 split_mode: str = "uneven",
                 sevir_catalog: Union[str, pd.DataFrame] = None,
                 sevir_data_dir: str = None,
                 start_date: datetime.datetime = None,
                 end_date: datetime.datetime = None,
                 datetime_filter = None,
                 catalog_filter = "default",
                 shuffle: bool = False,
                 shuffle_seed: int = 1,
                 output_type = np.float32,
                 preprocess: bool = True,
                 rescale_method: str = "01",
                 verbose: bool = False):
        super(SEVIRTorchDataset, self).__init__()
        self.layout = layout
        self.img_size = img_size
        self.sevir_dataloader = SEVIRDataLoader(
            dataset_dir=dataset_dir,
            data_types=["vil", ],
            seq_len=seq_len,
            raw_seq_len=raw_seq_len,
            sample_mode=sample_mode,
            stride=stride,
            batch_size=batch_size,
            layout=layout,
            num_shard=num_shard,
            rank=rank,
            split_mode=split_mode,
            sevir_catalog=sevir_catalog,
            sevir_data_dir=sevir_data_dir,
            start_date=start_date,
            end_date=end_date,
            datetime_filter=datetime_filter,
            catalog_filter=catalog_filter,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            output_type=output_type,
            preprocess=preprocess,
            rescale_method=rescale_method,
            downsample_dict=None,
            verbose=verbose)

        
        self.transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        # transforms.ToTensor(),
                        # trans.Lambda(lambda x: x/255.0),
                        # transforms.Normalize(mean=[0.5], std=[0.5]),
                        # trans.RandomCrop(data_config["img_size"]),

                    ])
    def __getitem__(self, index):
        data_dict = self.sevir_dataloader._idx_sample(index=index)
        data = data_dict["vil"]
        data = self.transform(data).unsqueeze(2)
        # print(data.shape)
        return data

    def __len__(self):
        return self.sevir_dataloader.__len__()

    def collate_fn(self, data_dict_list):
        r"""
        Parameters
        ----------
        data_dict_list:  list[Dict[str, torch.Tensor]]

        Returns
        -------
        merged_data: Dict[str, torch.Tensor]
            batch_size = len(data_dict_list) * data_dict["key"].batch_size
        """
        batch_dim = self.layout.find('N')
        data_list_dict = {
            key: [data_dict[key]
                  for data_dict in data_dict_list]
            for key in data_dict_list[0]}
        # TODO: key "mask" is not handled. Temporally fine since this func is not used
        data_list_dict.pop("mask", None)
        merged_dict = {
            key: torch.cat(data_list,
                           dim=batch_dim)
            for key, data_list in data_list_dict.items()}
        merged_dict["mask"] = None
        return merged_dict

    def get_torch_dataloader(self,
                             outer_batch_size=1,
                             collate_fn=None,
                             num_workers=1):
        # TODO: num_workers > 1
        r"""
        We set the batch_size in Dataset by default, so outer_batch_size should be 1.
        In this case, not using `collate_fn` can save time.
        """
        if outer_batch_size == 1:
            collate_fn = lambda x:x[0]
        else:
            if collate_fn is None:
                collate_fn = self.collate_fn
        dataloader = DataLoader(
            dataset=self,
            batch_size=outer_batch_size,
            collate_fn=collate_fn,
            pin_memory=False,
            num_workers=num_workers)
        return dataloader





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

HMF_COLORS = np.array([
    [82, 82, 82],
    [252, 141, 89],
    [255, 255, 191],
    [145, 191, 219]
]) / 255
PIXEL_SCALE = 255.0
BOUNDS = [0.0, 16.0, 31.0, 59.0, 74.0, 100.0, 133.0, 160.0, 181.0, 219.0, PIXEL_SCALE]
THRESHOLDS = (16, 74, 133, 160, 181, 219)


def gray2color(image, **kwargs):

    # 定义颜色映射和边界
    cmap = colors.ListedColormap(COLOR_MAP )
    bounds = BOUNDS
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # 将图像进行染色
    colored_image = cmap(norm(image))

    return colored_image


if __name__=='__main__':
        
    data = torch.randn(1, 256,256) * 255
    data = data.numpy()
    color = gray2color(data)



# if __name__ == '__main__':
#     dataset = SEVIRTorchDataset('sevir')
#     iterator = iter(dataset)
#     sample1 = next(iterator)
#     sample2 = next(iterator)

#     print(sample2.max(), sample2.min())
#     print(sample1.max(), sample1.min())
    
#     vis_res(sample1.numpy(), sample2.numpy(), save_path='test', save_grays=True, do_hmf=True, color_residual=True, save_colored=True)

#     print(len(dataset))