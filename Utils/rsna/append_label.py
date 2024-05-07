import os
from os.path import sep

from PIL import Image

dir = 'TODO'
new_dir = 'TODO'

label1 = 'PNEUMONIA'


for idx, dirname in enumerate(os.listdir(dir)):
    subdir = os.path.join(dir, dirname)
    new_subdir = os.path.join(new_dir, dirname)

    if not os.path.exists(new_subdir):
        os.makedirs(new_subdir)

    for labeldir in os.listdir(subdir):
        innerdir = os.path.join(subdir, labeldir)
        new_inner = os.path.join(new_subdir, labeldir)

        label = int(labeldir == label1)  # pneumonia is 1, normal is 0

        for filename in os.listdir(innerdir):
            old = os.path.join(innerdir, filename)

            file = filename.split('.')[0]
            filename = f'label{label}__{file}.png'
            new_filename = os.path.join(new_subdir, filename)

            img = Image.open(old)
            img = img.resize((128, 128))
            img.save(new_filename)