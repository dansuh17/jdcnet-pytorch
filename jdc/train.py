"""
Train script for
"Joint Detection and Classification of Singing Voice Melody
Using Convolutional Recurrent Neural Networks" by Kum et al. (2019)
"""
from typing import Union, Tuple
import json

import torch
from torch import nn
from torch.utils.data import Dataset
from torchland.trainer import NetworkTrainer, AttributeHolder, ModelInfo, TrainStage
from torchland.datasets.loader_builder import DefaultDataLoaderBuilder
from .dataset import SpecHz, MedleyDBMelodyDataset
from .model import JDCNet
from .loss import CrossEntropyLossWithGaussianSmoothedLabels


class MedleyDBDataLoaderBuilder(DefaultDataLoaderBuilder):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)


class JDCTrainer(NetworkTrainer):
    def __init__(self, config: dict):
        super().__init__(epoch=100, log_every_local=1, save_histogram=True)
        self.detection_weight = self.get_or_else(
            config, 'detection_weight', default_value=0.5)
        self.num_class = self.get_or_else(
            config, 'num_class', default_value=722)
        self.data_root = self.get_or_else(
            config, 'data_root', default_value='./data_in/medleydb_melody_jdc/medleydb_melody')
        self.batch_size = self.get_or_else(
            config, 'batch_size', default_value=64)
        self.num_workers = self.get_or_else(
            config, 'num_workers', default_value=4)
        self.lr_init = self.get_or_else(
            config, 'lr_init', default_value=1e-5)

        # setup
        jdc_model = JDCNet()
        input_size = (1, 31, 513)
        self.add_model('jdc_net', jdc_model, input_size, metric='loss')

        dataset = MedleyDBMelodyDataset(self.data_root)
        self.set_dataloader_builder(MedleyDBDataLoaderBuilder(
            dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers))
        self.add_criterion('loss_detection', nn.CrossEntropyLoss())
        self.add_criterion('loss_classification', CrossEntropyLossWithGaussianSmoothedLabels())

        adam = torch.optim.Adam(jdc_model.parameters(), lr=self.lr_init, weight_decay=1e-4)
        self.add_optimizer('adam', adam)

    @staticmethod
    def get_or_else(config: dict, key: str, default_value):
        if key in config:
            val = config[key]
        else:
            val = default_value
        print(f'Config - {key}: {val}')
        return val

    @staticmethod
    def make_performance_metric(input_: torch.Tensor, output, loss) -> dict:
        total_loss, classification_loss, detection_loss = loss
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'detection_loss': detection_loss,
        }

    def run_step(self, models: AttributeHolder[ModelInfo],
                 criteria: AttributeHolder[nn.Module],
                 optimizers,
                 input_: Union[torch.Tensor, Tuple[torch.Tensor]],
                 train_stage: TrainStage, *args, **kwargs):
        input_spec, target_labels, target_isvoice = input_
        # dimensions: (b, 1, 31, 513)
        input_spec = input_spec.unsqueeze(dim=1)  # add an axis to feature dimension

        model = models.jdc_net.model
        out_classification, out_detection = model(input_spec)
        classification_loss = criteria.loss_classification(out_classification, target_labels)

        # (b, 31, 2) => (b, 2, 31)
        out_detection = out_detection.transpose(1, 2)
        detection_loss = criteria.loss_detection(out_detection, target_isvoice)

        total_loss = classification_loss + self.detection_weight * detection_loss

        adam = optimizers.adam

        if train_stage == TrainStage.TRAIN:
            adam.zero_grad()
            total_loss.backward()

            # clip gradients to prevent gradient explosion for LSTM modules
            # torch.nn.utils.clip_grad_norm_(model.module.bilstm_classifier.parameters(), max_norm=0.25)
            # torch.nn.utils.clip_grad_norm_(model.module.bilstm_detector.parameters(), max_norm=0.25)
            torch.nn.utils.clip_grad_value_(model.module.parameters(), clip_value=1.0)
            adam.step()

        return (out_classification, out_detection), (total_loss, classification_loss, detection_loss)


if __name__ == '__main__':
    with open('config.json', 'r') as jsonf:
        config = json.load(jsonf)
    jdc_trainer = JDCTrainer(config)
    print(jdc_trainer._models.jdc_net.model.module.bilstm_classifier.parameters())
    print(jdc_trainer._models.jdc_net.model.module.parameters())
    # jdc_trainer.fit()
