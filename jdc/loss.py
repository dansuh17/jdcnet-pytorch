import torch
from torch import nn
import math
from .util import to_onehot, empty_onehot


class CrossEntropyLossWithGaussianSmoothedLabels(nn.Module):
    """
    See Also: https://github.com/pytorch/pytorch/issues/7455
    """
    def __init__(self, num_classes=722, blur_range=3):
        super().__init__()
        self.dim = -1
        self.num_classes = num_classes
        self.blur_range = blur_range

        # pre-calculate decayed values following Gaussian distribution
        # up to distance of three (== blur_range)
        self.gaussian_decays = [self.gaussian_val(dist=d) for d in range(blur_range + 1)]

    @staticmethod
    def gaussian_val(dist: int, sigma=1):
        return math.exp(-math.pow(2, dist) / (2 * math.pow(2, sigma)))

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        # pred: (b, 31, 722)
        # target: (b, 31)

        pred_logit = torch.log_softmax(pred, dim=self.dim)

        # out: (b, 31, 722)
        target_smoothed = self.smoothed_label(target)

        # calculate the 'cross entropy' for each of 31 features
        # TODO: use torch.nll_loss() instead?
        """
        pred: (b, 722, 31)
        target: (b, 31)
        nll_loss(input=pred, target=target, reduction='mean')
        """
        target_loss_sum = torch.sum(-pred_logit * target_smoothed, dim=self.dim)
        return torch.mean(target_loss_sum)  # and then take their mean

    def smoothed_label(self, target: torch.Tensor):
        # out: (b, 31, 722)
        target_onehot = empty_onehot(target, self.num_classes).to(target.device)

        # apply gaussian smoothing
        target_smoothed = self.gaussian_blur(target, target_onehot)
        target_smoothed = to_onehot(target, self.num_classes, target_smoothed)
        return target_smoothed

    def gaussian_blur(self, target: torch.Tensor, one_hot: torch.Tensor):
        # blur the one-hot vector with gaussian decay
        with torch.no_grad():
            # Going in the reverse direction from 3 -> 0 since the value on the clamped index
            # will override the previous value
            # when the class index is less than 4 or greater then (num_class - 4).
            for dist in range(self.blur_range, -1, -1):
                one_hot = self.set_decayed_values(dist, target, one_hot)
        return one_hot

    def set_decayed_values(self, dist: int, target_idx: torch.Tensor, one_hot: torch.Tensor):
        # size of target_idx: (batch, num_seq) = (batch, 31)
        # size of one_hot: (batch, num_seq, num_classes) = (batch, 31, 722)
        for direction in [1, -1]:  # apply at both positive / negative directions
            # used `clamp` to prevent index from underflowing / overflowing
            blur_idx = torch.clamp(
                target_idx + (direction * dist), min=0, max=self.num_classes - 1)
            # set decayed values at indices represented by blur_idx
            decayed_val = self.gaussian_decays[dist]
            one_hot = one_hot.scatter_(
                dim=2, index=torch.unsqueeze(blur_idx, dim=2), value=decayed_val)
        return one_hot


def test_gaussian_blur():
    """
    Before blurring:

    tensor([[[0., 0., 0., 0., 1., 0., 0.],
             [0., 0., 1., 0., 0., 0., 0.],
             [1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0., 0.]]])

    After blurring:

    tensor([[[0.0000, 0.1353, 0.3679, 0.6065, 1.0000, 0.6065, 0.3679],
             [0.3679, 0.6065, 1.0000, 0.6065, 0.3679, 0.1353, 0.0000],
             [1.0000, 0.6065, 0.3679, 0.1353, 0.0000, 0.0000, 0.0000],
             [0.0000, 0.1353, 0.3679, 0.6065, 1.0000, 0.6065, 0.3679]]])
    """
    loss = CrossEntropyLossWithGaussianSmoothedLabels(num_classes=7)
    print(loss.smoothed_label(torch.LongTensor(1, 4) % 7))


if __name__ == '__main__':
    test_gaussian_blur()
