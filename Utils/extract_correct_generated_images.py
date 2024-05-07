import os
import shutil
from dataclasses import dataclass

images = 'Images'
rgb = 'ImagesRGB'

correct = 'Correct'
padded = 'TestValPadded'

target_count = 2048

train = 'TRAIN'
test = 'TEST'
val = 'VAL'


dirs = [
    images,
    rgb,
]

type_dirs = [
    train,
    test,
    val,
]


@dataclass
class Stats:
    total_files: int
    incorrect_files: int


def text(stats_by_dir):
    lines = [
        f'Dir: {dir} ---- Total: {stats.total_files} -- Incorrect: {stats.incorrect_files} --> Correct %: {1 - (stats.incorrect_files / stats.total_files)}' for dir, stats in stats_by_dir.items()
    ]

    return '/n'.join(lines)


def copy(dir, incorrect_name, pad):
    stats_by_dir = {}

    new_dir = f'{dir}{correct}'
    os.mkdir(new_dir)

    if pad:
        padded_dir = os.path.join(base_path, f'{rgb}{correct}{padded}')
        os.mkdir(padded_dir)

    for sub_dir in type_dirs:
        path = os.path.join(dir, sub_dir)

        new_path = os.path.join(new_dir, sub_dir)
        os.mkdir(new_path)

        if pad and sub_dir != train:
            new_pad_path = os.path.join(padded_dir, sub_dir)
            os.mkdir(new_pad_path)

        files = 0
        incorrect = 0

        for file in os.listdir(path):
            files += 1

            if incorrect_name in file:
                incorrect += 1
                continue

            old_file = os.path.join(path, file)
            new_file = os.path.join(new_path, file)
            shutil.copyfile(old_file, new_file)

            if pad and sub_dir != train:
                new_padded_file = os.path.join(new_pad_path, file)
                shutil.copyfile(old_file, new_padded_file)

        stats_by_dir[sub_dir] = Stats(total_files=files, incorrect_files=incorrect)

    return stats_by_dir


def pad(base_path):
    padded_dir = os.path.join(os.path.join(base_path, f'{rgb}{correct}{padded}'), test)

    file_count = len([None for file in os.listdir(padded_dir)])

    missing_files = target_count - file_count

    if missing_files <= 0:
        return

    padding_from_label_1 = int(missing_files / 2)
    val_dir = os.path.join(os.path.join(base_path, f'{rgb}{correct}{padded}'), val)

    val_files = sorted(os.listdir(val_dir))
    files_to_copy = val_files[:padding_from_label_1]

    still_missing_files = target_count - file_count - len(files_to_copy)
    assert still_missing_files == padding_from_label_1 or still_missing_files == padding_from_label_1 + 1  # split (almost) in the middle

    files_to_copy.extend(val_files[-still_missing_files:])

    for file in files_to_copy:
        old_file = os.path.join(val_dir, file)
        new_file = os.path.join(padded_dir, file)
        shutil.copyfile(old_file, new_file)

    new_file_count = len([1 for file in os.listdir(padded_dir)])

    assert new_file_count == target_count

    shutil.rmtree(val_dir, ignore_errors=True)


def run(base_path, incorrect_name):
    stats_by_dir_by_dir = []

    for dir in dirs:
        full_path = os.path.join(base_path, dir)
        stats_by_dir = copy(full_path, incorrect_name, dir == rgb)
        stats_by_dir_by_dir.append(stats_by_dir)

    assert len(stats_by_dir_by_dir) == 2
    assert stats_by_dir_by_dir[0] == stats_by_dir_by_dir[1]

    pad(base_path)

    print(f'Results for {base_path}:')
    print(f'Filtering: {incorrect_name}')
    print(text(stats_by_dir_by_dir[0]))
    print('-----------------------------------------')


if __name__ == '__main__':
    base_paths = [

        ('TODO', 'COUNTER_'),

    ]

    for base_path, incorrect_name in base_paths:
        run(base_path, incorrect_name)