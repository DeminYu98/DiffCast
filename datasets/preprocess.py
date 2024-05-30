import numpy as np

import matplotlib.pyplot as plt
import os
import h5py 
from tqdm import tqdm
from PIL import Image

TRAIN_TEST_SPLIT = 0.85
THRESHOLD = 3.0
data_dir = "/data/to/path/nh_radar_comp_echo"

all_dirs = sorted(os.listdir(data_dir))

train_dirs = all_dirs[:int(len(all_dirs)*TRAIN_TEST_SPLIT)]
test_dirs = all_dirs[int(len(all_dirs)*TRAIN_TEST_SPLIT):]
print(len(train_dirs), len(test_dirs))

h5_data = "data/to/path/shanghai.h5"

h5_file = h5py.File(h5_data, 'w')
h5_file.create_group('train')
h5_file.create_group('test')

seq_len = 0
min_vals = 30000
min_key = ""

for dir in tqdm(train_dirs):
    img_paths = os.listdir(os.path.join(data_dir,dir))
    long_seq = []

    for path in sorted(img_paths):
        img_path = os.path.join(data_dir, dir, path)
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        image_array = np.array(image)
        long_seq.append(image_array)

    i = 10
    
    while i + 20 < len(long_seq):
        if long_seq[i].mean() > THRESHOLD:        #THRESHOLD
            seq = long_seq[i-5:i+20]
            all_means = sum([frame.mean() for frame in seq])
            # if all_means > 3.5 * 30:
            key = str(seq_len)
            if all_means < min_vals:
                min_vals = all_means
                min_key = key
            h5_file['train'].create_dataset(str(key), data=seq, dtype='uint8', compression='lzf')
            i = i + 15
            seq_len += 1
            continue
        i = i + 5

h5_file['train'].create_dataset('all_len', data=seq_len)

print(seq_len)
print(min_vals)
print(min_key)

seq_len = 0
min_vals = 30000
min_key = ""

for dir in tqdm(test_dirs):
    img_paths = os.listdir(os.path.join(data_dir,dir))
    long_seq = []

    for path in sorted(img_paths):
        img_path = os.path.join(data_dir, dir, path)
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        image_array = np.array(image) 
        long_seq.append(image_array)

    i = 10
    
    while i + 20 < len(long_seq):
        if long_seq[i].mean() > THRESHOLD:
            seq = long_seq[i-5:i+20]
            all_means = sum([frame.mean() for frame in seq])
            # if all_means > 3.5 * 30:
            key = str(seq_len)
            if all_means < min_vals:
                min_vals = all_means
                min_key = key
            h5_file['test'].create_dataset(str(key), data=seq, dtype='uint8', compression='lzf')
            i = i + 5
            seq_len += 1
            continue
        i = i + 5

h5_file['test'].create_dataset('all_len', data=seq_len)

h5_file.close()
print(seq_len)
print(min_vals)
print(min_key)




