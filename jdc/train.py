import math
import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import MedleyDBMelodyDataset, SpecHz
from .model import JDCNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_epoch = 200  # ?
batch_size = 5  # ?
num_workers = 8


class CrossEntropyLossWithGaussianSmoothedLabels(nn.Module):
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
        # target: (b, 31)
        pred_logit = torch.log_softmax(pred, dim=self.dim)

        # out: (b, 31, 722)
        target_smoothed = self.smoothed_label(target)

        # calculate the cross entropy loss
        return torch.mean(torch.sum(- pred_logit * target_smoothed, dim=self.dim))

    def smoothed_label(self, target: torch.Tensor):
        # out: (b, 31, 722)
        target_onehot = torch.FloatTensor(*self.onehot_size(target)).zero_().to(device)  # TODO: better way?

        # apply gaussian smoothing
        target_smoothed = self.gaussian_blur(target, target_onehot)
        target_smoothed = self.to_one_hot(target, target_smoothed)
        return target_smoothed

    def onehot_size(self, target: torch.Tensor):
        # calculates the size of the one-hot vector
        return target.size() + (self.num_classes, )

    def to_one_hot(self, target: torch.Tensor, one_hot: torch.Tensor):
        # creates a one hot vector provided the target indices
        # and the Tensor that holds the one-hot vector
        with torch.no_grad():
            one_hot = one_hot.scatter_(
                dim=2, index=torch.unsqueeze(target, dim=2), value=1.0)
        return one_hot

    def gaussian_blur(self, target: torch.Tensor, one_hot: torch.Tensor):
        # blur the one-hot vector with gaussian decay
        with torch.no_grad():
            # Going in the reverse direction from 3 -> 0 since the value on the clamped index
            # will override the previous value
            # when the class index is less than 4 or greater then (num_class - 4).
            for dist in range(self.blur_range, -1, -1):
                # distance in the negative direction
                # used `clamp` to prevent index from underflowing / overflowing
                blur_idx = torch.clamp(target - dist, min=0, max=self.num_classes - 1)
                decayed_val = self.gaussian_decays[dist]
                one_hot = one_hot.scatter_(
                    dim=2, index=torch.unsqueeze(blur_idx, dim=2), value=decayed_val)

                # distance in the positive direction
                blur_idx = torch.clamp(target + dist, min=0, max=self.num_classes - 1)
                decayed_val = self.gaussian_decays[dist]
                one_hot = one_hot.scatter_(
                    dim=2, index=torch.unsqueeze(blur_idx, dim=2), value=decayed_val)
        return one_hot


loss_classification = CrossEntropyLossWithGaussianSmoothedLabels(blur_range=0)
loss_detection = nn.CrossEntropyLoss()


def fit():
    # spread the calculations over distributed cores
    model = torch.nn.parallel.DataParallel(JDCNet().to(device))

    dataset = MedleyDBMelodyDataset(root='./out_root')
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=True, pin_memory=True)

    # total_loss = loss_classification(classified_res, gaussian_blur(target_pitch)) + loss_detection(detection_res, target_isvoice)

    num_classes = 722
    num_classes_isvoice = 2
    seq_size = 31  # 31 frames

    for epoch in range(total_epoch):
        for input_ in dataloader:
            input_spec, target_labels, target_isvoice = input_
            input_spec = input_spec.to(device).unsqueeze(dim=1)  # add an axis to feature dimension
            # size: (b, 31)
            target_labels = target_labels.to(device)
            # size: (b, 31)
            target_isvoice = target_isvoice.to(device)  # TODO: implement detection loss

            out_classification, out_detection = model(input_spec)
            # out_labels, out_isvoice = torch.split(out_classification, [1, 721], dim=2)

            classification_loss = loss_classification(out_classification, target_labels)
            print(classification_loss)


if __name__ == '__main__':
    fit()

    # TESTING GAUSSIAN BLUR
    # loss = CrossEntropyLossWithGaussianSmoothedLabels(num_classes=7)
    # print(loss.smoothed_label(torch.LongTensor(1, 4) % 7))
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

