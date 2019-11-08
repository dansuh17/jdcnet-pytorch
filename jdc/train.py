"""
Train script for
"Joint Detection and Classification of Singing Voice Melody
Using Convolutional Recurrent Neural Networks" by Kum et al. (2019)
"""
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from torchland.trainer.trainer import NetworkTrainer, TrainStage
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
        # optional value with default value
        val = config[key] if key in config else default_value
        print(f'Config - {key}: {val}')
        return val

    @staticmethod
    def make_performance_metric(input_: torch.Tensor, output, loss) -> dict:
        total_loss, classification_loss, detection_loss = loss
        # doing this will show the printed log / graphs of these values
        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'detection_loss': detection_loss,
        }

    def run_step(
            self, models, criteria, optimizers, input_, train_stage: TrainStage,
            *args, **kwargs):
        input_spec, target_labels, target_isvoice = input_
        # dimensions: (b, 1, 31, 513)
        input_spec = input_spec.unsqueeze(dim=1)  # add an axis to feature dimension

        model = models.jdc_net.model

        # forward-propagate
        out_classification, out_detection = model(input_spec)

        # loss for pitch class classification
        classification_loss = criteria.loss_classification(out_classification, target_labels)

        # loss for voice detection = 'isvoice?'
        # (b, 31, 2) => (b, 2, 31)
        out_detection = out_detection.transpose(1, 2)
        detection_loss = criteria.loss_detection(out_detection, target_isvoice)

        # take the weighted sum of the two losses
        total_loss = classification_loss + self.detection_weight * detection_loss

        if train_stage == TrainStage.TRAIN:
            adam = optimizers.adam
            adam.zero_grad()
            total_loss.backward()

            # clip gradients to prevent gradient explosion for LSTM modules
            torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_norm=0.5)
            adam.step()

        return (out_classification, out_detection), (total_loss, classification_loss, detection_loss)

    def post_step(
            self, input, output, metric: dict,
            dataset_size: int, train_stage: TrainStage):
        super().post_step(input, output, metric, dataset_size, train_stage)

        if self._local_step % 500 != 0:
            return

        # log the melody contour
        _, target_labels, target_isvoice = input

        # retrieve inputs
        # target_labels: (b, num_frames)
        batch_size, num_frames = target_labels.size()

        # retrieve outputs
        out_classification = output[0]
        # find the predicted pitch class by taking the argmax of classification output
        # out_classification: (b, num_frames, num_classes)
        _, pred_classes = out_classification.max(dim=2)

        # log the ratio of 'isvoice'
        numel = float(target_isvoice.numel())
        isvoice_ratio = float(target_isvoice.sum().data) / numel
        self._writer.add_scalar('isvoice_ratio', isvoice_ratio, self._global_step)

        # log the predicted /
        default_numsamps = 3  # use default of three examples
        num_samples = min(batch_size, default_numsamps)
        for samp_idx in range(num_samples):
            for t in range(num_frames):
                tag_name = f'melody_e{self._epoch}_s{self._global_step}_i{samp_idx}'
                # log the ground truth
                self._writer.add_scalar(
                    f'{tag_name}/truth', target_labels[samp_idx][t], global_step=t)

                # log the predicted result
                self._writer.add_scalar(
                    f'{tag_name}/predicted', pred_classes[samp_idx][t], global_step=t)


if __name__ == '__main__':
    with open('config.json', 'r') as jsonf:
        config = json.load(jsonf)
    jdc_trainer = JDCTrainer(config)
    jdc_trainer.fit()
