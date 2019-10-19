import torch


def empty_onehot(target: torch.Tensor, num_classes: int):
    # target_size = (batch, dim1, dim2)
    # one_hot size = (batch, dim1, dim2, num_classes)
    onehot_size = target.size() + (num_classes, )
    return torch.FloatTensor(*onehot_size).zero_()


def to_onehot(target: torch.Tensor, num_classes: int, src_onehot: torch.Tensor = None):
    if src_onehot is None:
        one_hot = empty_onehot(target, num_classes)
    else:
        one_hot = src_onehot

    last_dim = len(one_hot.size()) - 1

    # creates a one hot vector provided the target indices
    # and the Tensor that holds the one-hot vector
    with torch.no_grad():
        one_hot = one_hot.scatter_(
            dim=last_dim, index=torch.unsqueeze(target, dim=last_dim), value=1.0)
    return one_hot
