import os
from PIL import Image

base_path = 'TODO'
rgb_base_path = 'TODO'

train = f'TRAIN'
val = f'VAL'
test = f'TEST'

dirs = [test]

for dir in dirs:
    old_dir_path = os.path.join(base_path, dir)
    new_dir_path = os.path.join(rgb_base_path, dir)

    os.makedirs(new_dir_path)

    for file in os.listdir(old_dir_path):
        complete_filename = os.path.join(old_dir_path, file)
        new_filename = os.path.join(new_dir_path, file)

        img = Image.open(complete_filename).resize((128, 128))
        img.save(new_filename)