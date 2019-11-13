#!/usr/bin/env python3
"""
Script for training the neural network described in:

Kum et al. - "Joint Detection and Classification of Singing Voice Melody Using
Convolutional Recurrent Neural Networks" (2019)

a.k.a "JDCNet"
"""
import argparse
import json
from jdc.train import JDCTrainer

parser = argparse.ArgumentParser(description='Train JDCNet')
parser.add_argument('-c', '--config', type=str, default='config.json', help='config file')
args = parser.parse_args()

# read configuration file and start training
config_file = args.config
with open(config_file, 'r') as jsonf:
    config = json.load(jsonf)
jdc_trainer = JDCTrainer(config)
jdc_trainer.fit()
