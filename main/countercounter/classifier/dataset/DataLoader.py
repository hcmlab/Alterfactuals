GRAY_MEAN_FACTOR = 0.5822
GRAY_STD_FACTOR = 0.1572


def denormalize(data):
    return (data * 0.5) + 0.5


def normalize_for_resnet(data):
    data = denormalize(data)
    return (data - GRAY_MEAN_FACTOR) / GRAY_STD_FACTOR