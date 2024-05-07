import os

from PIL import Image

dir = 'TODO'
new_dir = 'TODO'

for idx, dir_name in enumerate(os.listdir(dir)):
    type_dir = os.path.join(dir, dir_name)
    new_type_dir = os.path.join(new_dir, dir_name)

    for jdx, subdir_name in enumerate(os.listdir(type_dir)):
        complete_dir_name = os.path.join(type_dir, subdir_name)
        new_dir_name = os.path.join(new_type_dir, subdir_name)

        if not os.path.exists(new_dir_name):
            os.makedirs(new_dir_name)

        for filename in os.listdir(complete_dir_name):
            old = os.path.join(complete_dir_name, filename)
            new = os.path.join(new_dir_name, filename)

            img = Image.open(old).convert('L')
            img.save(new)
