import torch
from torch import nn

loss_classification = nn.CrossEntropyLoss()
loss_detection = nn.CrossEntropyLoss()

# TODO
target_pitch = None
target_isvoice = None
classified_res = None
detection_res = None


def gaussian_blur(one_hot):
    return None


total_loss = loss_classification(classified_res, gaussian_blur(target_pitch)) + loss_detection(detection_res, target_isvoice)


if __name__ == '__main__':
    pass
