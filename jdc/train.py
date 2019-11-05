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
from torchland.trainer.trainer import NetworkTrainer, AttributeHolder, ModelInfo, TrainStage
from torchland.datasets.loader_builder import DefaultDataLoaderBuilder
from .dataset import SpecHz, MedleyDBMelodyDataset
from .model import JDCNet
from .loss import CrossEntropyLossWithGaussianSmoothedLabels


class MedleyDBDataLoaderBuilder(DefaultDataLoaderBuilder):
    def __init__(self, dataset: Dataset, *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)


class JDCTrainer(NetworkTrainer):
    def __init__(self, config: dict):
        super().__init__(epoch=self.get_or_else(config, 'epoch', 100),
                         log_every_local=self.get_or_else(config, 'log_every_local', 20))
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
            config, 'lr_init', default_value=3e-4)
        self.lr_decay_step = self.get_or_else(
            config, 'lr_decay_step', default_value=40)

        # setup
        jdc_model = JDCNet()
        input_size = (1, 31, 513)
        self.add_model('jdc_net', jdc_model, input_size, metric='loss')

        dataset = MedleyDBMelodyDataset(self.data_root)
        self.set_dataloaders(MedleyDBDataLoaderBuilder(
            dataset=dataset, batch_size=self.batch_size, num_workers=self.num_workers))
        self.add_criterion('loss_detection', nn.CrossEntropyLoss())
        self.add_criterion('loss_classification', CrossEntropyLossWithGaussianSmoothedLabels())

        adam = torch.optim.Adam(jdc_model.parameters(), lr=self.lr_init, weight_decay=1e-4)
        self.add_optimizer('adam', adam)
        adam_steplr = torch.optim.lr_scheduler.StepLR(adam, step_size=40)
        self.add_lr_scheduler('steplr', adam_steplr)

    @staticmethod
    def get_or_else(config: dict, key: str, default_value):
        val = config[key] if key in config else default_value
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

    def post_step(
            self, input, output, metric: dict,
            dataset_size: int, train_stage: TrainStage):
        super().post_step(input, output, metric, dataset_size, train_stage)

        if self._local_step % self._log_every_local == 0:
            _, target_labels, target_isvoice = input
            numel = float(target_isvoice.numel())

            # log the ratio of 'isvoice'
            isvoice_ratio = float(target_isvoice.sum().data) / numel
            self._writer.add_scalar('isvoice_ratio', isvoice_ratio, self._global_step)

            # (b, num_frames)
            batch_size = target_labels.size()[0]
            num_frames = target_labels.size()[1]
            for samp_idx in range(3):
                # (1, 31, 722)
                one_hot = torch.FloatTensor(1, num_frames, 722).zero_().to(self._device)
                # labels: (1, 31, 1)
                labels = torch.unsqueeze(torch.unsqueeze(target_labels[samp_idx], dim=0), dim=2)
                # one_hot: (1, 31, 722) => (1, 722, 31)
                one_hot = one_hot.scatter_(dim=2, index=labels, value=1.0).transpose(1, 2)
                self._writer.add_image(
                    f'image_{self._epoch}/s{samp_idx}',
                    one_hot,
                    global_step=self._global_step)

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
            torch.nn.utils.clip_grad_value_(model.module.parameters(), clip_value=0.25)
            adam.step()

        return (out_classification, out_detection), (total_loss, classification_loss, detection_loss)


if __name__ == '__main__':
    with open('config.json', 'r') as jsonf:
        config = json.load(jsonf)
    jdc_trainer = JDCTrainer(config)
    jdc_trainer.fit()
