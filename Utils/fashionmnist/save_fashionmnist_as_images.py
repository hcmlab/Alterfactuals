import random
import uuid
from os.path import sep

from torchvision import datasets

mnist_train = datasets.FashionMNIST(root='/files/', train=True, download=True, transform=None)
mnist_test = datasets.FashionMNIST(root='/files/', train=False, download=True, transform=None)

print(f'Original train len: {len(mnist_train)}')
print(f'Original test len: {len(mnist_test)}')

train_data = []
test_data = []

loaders = [(mnist_train, train_data), (mnist_test, test_data)]

# 0 tshirt 1 trouser 2 pullover 3 dress 4 coat 5 sandal 6 shirt 7 sneaker 8 bag 9 ankle boot
rel_labels = [2, 6]

for loader, data in loaders:
    for img, label in loader:
        if label in rel_labels:
            data.append((img, label))

random.shuffle(train_data)
val_size = int(len(train_data) * 0.1)

train_data = train_data[val_size:]
val_data = train_data[:val_size]

print(f'Train len: {len(train_data)}')
print(f'Val len: {len(val_data)}')
print(f'Test len: {len(test_data)}')

base_path = 'TODO'
train = f'TRAIN{sep}'
val = f'VAL{sep}'
test = f'TEST{sep}'

data_files = [
    (train_data, train),
    (test_data, test),
    (val_data, val)
]

for data, filename in data_files:
    dir_path = base_path + filename

    for img, label in data:
        name = uuid.uuid4().hex

        full_filename = dir_path + f'label{label}__{name}.png'
        img.save(full_filename)
