import os
import random
from os.path import sep

import pandas as pd
import pydicom as dicom
from PIL import Image

csv_path = 'TODO'
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

df = pd.read_csv(csv_path)

for idx, filename in enumerate(os.listdir(base_path)):
    id = filename.split('.')[0]
    label = df[df['patientId'] == id]['Target'].values[0]

    file = base_path + f'{sep}' + filename
    new_filename = f'{id}.png'
    img = dicom.dcmread(file)
    img = Image.fromarray(img.pixel_array)
    img = img.resize((128, 128))

    if idx in train_idxs:
        save_path = base_path + f'{sep}{train}label{label}__{new_filename}'
    elif idx in test_idxs:
        save_path = base_path + f'{sep}{test}label{label}__{new_filename}'
    elif idx in val_idxs:
        save_path = base_path + f'{sep}{val}label{label}__{new_filename}'
    else:
        raise ValueError(f'Idx {idx} not in any data set partition!')

    img.save(save_path)
    print(idx)

