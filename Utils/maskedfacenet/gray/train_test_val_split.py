import os
import random
from os.path import sep

from PIL import Image

base_path = 'TODO'
train = f'TRAIN{sep}'
val = f'VAL{sep}'
test = f'TEST{sep}'

file_idxs = list(range(0, len(os.listdir(base_path))))

random.shuffle(file_idxs)

test_size = int(len(file_idxs) * 0.1)

test_idxs = set(file_idxs[:test_size])
val_idxs = set(file_idxs[test_size:2*test_size])
train_idxs = (file_idxs[2*test_size:])

for idx, filename in enumerate(os.listdir(base_path)):
    file = base_path + '{sep}' + filename
    img = Image.open(file)

    if idx in train_idxs:
        save_path = base_path + f'{sep}{train}{filename}'
    elif idx in test_idxs:
        save_path = base_path + f'{sep}{test}{filename}'
    elif idx in val_idxs:
        save_path = base_path + f'{sep}{val}{filename}'
    else:
        raise ValueError(f'Idx {idx} not in any data set partition!')

    img.save(save_path)
    print(idx)

