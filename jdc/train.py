"""
Train script for
"Joint Detection and Classification of Singing Voice Melody
Using Convolutional Recurrent Neural Networks" by Kum et al. (2019)
"""
from typing import Union, Tuple

import torch
from torch import nn
from torchland.trainer import NetworkTrainer, AttributeHolder, ModelInfo, TrainStage
from .dataset import SpecHz
from .medleydb_dataloader import MedleyDBDataLoaderBuilder
from .model import JDCNet
from .loss import CrossEntropyLossWithGaussianSmoothedLabels


class JDCTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__(epoch=100)
        self.detection_weight = 0.5
        self.num_class = 722
        self.data_root = './out_root'
        #self.batch_size = 64
        self.batch_size = 5
        self.num_workers = 8
        self.lr_init = 3e-4

        # setup
        jdc_model = JDCNet()
        input_size = (1, 31, 513)
        self.add_model('jdc_net', jdc_model, input_size, metric='loss')
        self.set_dataloader_builder(MedleyDBDataLoaderBuilder(
            data_root=self.data_root, batch_size=self.batch_size, num_workers=self.num_workers))
        self.add_criterion('loss_detection', nn.CrossEntropyLoss())
        self.add_criterion('loss_classification', CrossEntropyLossWithGaussianSmoothedLabels())
        self.add_optimizer('adam', torch.optim.Adam(jdc_model.parameters(), lr=self.lr_init))

    def run_step(self, models: AttributeHolder[ModelInfo],
                 criteria: AttributeHolder[nn.Module],
                 optimizers,
                 input_: Union[torch.Tensor, Tuple[torch.Tensor]],
                 train_stage: TrainStage, *args, **kwargs):
        input_spec, target_labels, target_isvoice = input_
        # dimensions: (b, 1, 31, 513)
        input_spec = input_spec.unsqueeze(dim=1)  # add an axis to feature dimension

        out_classification, out_detection = models.jdc_net.model(input_spec)
        classification_loss = criteria.loss_classification(out_classification, target_labels)

        # (b, 31, 2) => (b, 2, 31)
        out_detection = out_detection.transpose(1, 2)
        detection_loss = criteria.loss_detection(out_detection, target_isvoice)

        total_loss = classification_loss + self.detection_weight * detection_loss

        adam = optimizers.adam

        if train_stage == TrainStage.TRAIN:
            adam.zero_grad()
            total_loss.backward()
            adam.step()

        return (out_classification, out_detection), total_loss


if __name__ == '__main__':
    jdc_trainer = JDCTrainer()
    jdc_trainer.fit()
