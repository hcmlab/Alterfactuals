import os
from os.path import sep

from PIL import Image

base_path = 'TODO'
train = f'TRAIN{sep}'
val = f'VAL{sep}'
test = f'TEST{sep}'

for idx, filename in enumerate(os.listdir(base_path)):
    file = base_path + f'{sep}' + filename
    img = Image.open(file)
    img = img.resize((128, 128))
    img.save(f'{base_path}{sep}label{1}__{filename}')
    print(idx)

