import os
from os.path import sep

from PIL import Image

base_path = 'TODO'

crnt_counter = 0
dir_counter = 0

dir_name = f'{base_path}{sep}{dir_counter:03}'

for idx, filename in enumerate(os.listdir(base_path)):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    file = base_path + f'{sep}' + filename
    img = Image.open(file)
    img.save(dir_name + f'{sep}' + filename)
    print(idx)

    crnt_counter += 1

    if crnt_counter == 1000:
        crnt_counter = 0
        dir_counter += 1
        dir_name = f'{base_path}{sep}{dir_counter:03}'
