"""
Train script for
"Joint Detection and Classification of Singing Voice Melody
Using Convolutional Recurrent Neural Networks" by Kum et al. (2019)
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .dataset import MedleyDBMelodyDataset, SpecHz
from .model import JDCNet
from .loss import CrossEntropyLossWithGaussianSmoothedLabels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_class = 722
total_epoch = 50
batch_size = 5
num_workers = 8
detection_weight = 0.5
writer = SummaryWriter()

loss_classification = CrossEntropyLossWithGaussianSmoothedLabels()
loss_detection = nn.CrossEntropyLoss()

# spread the calculations over distributed cores
model = torch.nn.parallel.DataParallel(JDCNet().to(device))

optimizer = torch.optim.Adam(model.module.parameters(), lr=3e-4)

dataset = MedleyDBMelodyDataset(root='./out_root')
dataloader = DataLoader(
    dataset, batch_size=batch_size,
    shuffle=True, drop_last=True, pin_memory=True)

global_step = 0
for epoch in range(total_epoch):
    for input_ in dataloader:
        # parse inputs
        input_spec, target_labels, target_isvoice = input_
        # out: (b, 1, 31, 513)
        input_spec = input_spec.to(device).unsqueeze(dim=1)  # add an axis to feature dimension
        target_labels = target_labels.to(device)  # size: (b, 31)
        target_isvoice = target_isvoice.to(device)  # size: (b, 31)

        out_classification, out_detection = model(input_spec)
        classification_loss = loss_classification(out_classification, target_labels)

        # (b, 31, 2) => (b, 2, 31)
        out_detection = out_detection.transpose(1, 2)
        detection_loss = loss_detection(out_detection, target_isvoice)

        total_loss = classification_loss + detection_weight * detection_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f'epoch: {epoch:03d}, step: {global_step:04d}, '
              f'total_loss: {total_loss.data:04f}, class_loss: {classification_loss.data:04f}, '
              f'detection_loss: {detection_loss.data:04f}')
        writer.add_scalar('total_loss', total_loss.data, global_step)
        writer.add_scalar('class_loss', classification_loss.data, global_step)
        writer.add_scalar('detection_loss', detection_loss.data, global_step)

        global_step += 1
