import torch
from torch import nn
from torch.utils.data import DataLoader
from .dataset import MedleyDBMelodyDataset, SpecHz
from .model import JDCNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_epoch = 200 # ?
batch_size = 5  # ?
num_workers = 8

loss_classification = nn.CrossEntropyLoss()
loss_detection = nn.CrossEntropyLoss()

# TODO
target_pitch = None
target_isvoice = None
classified_res = None
detection_res = None


def gaussian_blur(one_hot):
    # TODO:
    return one_hot


model = torch.nn.parallel.DataParallel(JDCNet().to(device))

dataset = MedleyDBMelodyDataset(root='./out_root')
dataloader = DataLoader(
    dataset, batch_size=batch_size,
    shuffle=True, drop_last=True, pin_memory=True)


# total_loss = loss_classification(classified_res, gaussian_blur(target_pitch)) + loss_detection(detection_res, target_isvoice)

num_classes = 722
num_classes_isvoice = 2

for epoch in range(total_epoch):
    for input_ in dataloader:
        input_spec, target_labels, target_isvoice = input_
        input_spec = input_spec.to(device).unsqueeze(dim=1)  # add an axis to feature dimension
        target_labels = target_labels.to(device)
        target_isvoice = target_isvoice.to(device)
        print(input_spec.shape)

        out_classification, out_detection = model(input_spec)
        out_labels, out_isvoice = torch.split(out_classification, [1, 721], dim=2)
        print(target_labels.size())

        target_onehot = torch.FloatTensor(batch_size, num_classes).zero_().to(device)
        print(target_onehot.size())
        target_onehot = target_onehot.scatter(1, target_labels, 1)
        target_onehot = gaussian_blur(target_onehot)

        total_loss = loss_classification(out_labels, target_onehot) + loss_detection(out_detection, )


if __name__ == '__main__':
    pass
