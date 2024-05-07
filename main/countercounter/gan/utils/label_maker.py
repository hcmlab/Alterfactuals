import torch


def get_real_labels(batch_size, device, one_sided_label_smoothing=False):
    if one_sided_label_smoothing:
        t = torch.Tensor(batch_size).to(device)
        t.fill_(0.9)
        return t
    else:
        return torch.ones(batch_size).to(device)


def get_fake_labels(batch_size, device):
    return torch.zeros(batch_size).to(device)