import os
import shutil
from os.path import sep

dir = f'TODO'
TRAIN_DIR = dir + f'{sep}TRAIN'
TEST_DIR = dir + f'{sep}TEST'
VAL_DIR = dir + f'{sep}VAL'

test = {4, 10, 22, 24, 33, 42, 43, 57, 59, 68}
val = {7, 16, 32, 38, 44, 45, 55}

test_files = set()
val_files = set()
train_files = set()

for idx, filename in enumerate(os.listdir(dir)):
    if filename == 'TRAIN' or filename == 'TEST' or filename == 'VAL':
        continue

    complete_filename = os.path.join(dir, filename)

    name_parts = filename.split('_')
    id_number = name_parts[1]

    if id_number[0] == '0':
        iden = int(id_number[1:])
    else:
        iden = int(id_number)

    if iden in test:
        test_files.add((complete_filename, filename))
    elif iden in val:
        val_files.add((complete_filename, filename))
    else:
        train_files.add((complete_filename, filename))

for (test_file, filename) in test_files:
    new_filename = os.path.join(TEST_DIR, filename)
    shutil.move(test_file, new_filename)

for (test_file, filename) in train_files:
    new_filename = os.path.join(TRAIN_DIR, filename)
    shutil.move(test_file, new_filename)

for (test_file, filename) in val_files:
    new_filename = os.path.join(VAL_DIR, filename)
    shutil.move(test_file, new_filename)