#!/usr/bin/env python3
"""
Script for extracting melody from audio file using trained JDCNet.
"""
import argparse
import os
from jdc.inference import extract_melody, save_midi, read_jdc_model
from jdc.util import PitchLabeler

# define program arguments
parser = argparse.ArgumentParser(
    description='Extract Singing Voice Melody to MIDI from Audio File')
parser.add_argument('-i', '--input_audio', type=str, help='input audio file')
parser.add_argument('-o', '--out_dir', type=str, nargs='?',
                    default='melody_results',
                    help='path to directory where results are stored')
parser.add_argument('-n', '--name', type=str, nargs='?',
                    default='my_melody', help='melody name')
parser.add_argument('-m', '--model', type=str, help='path to model file')
args = parser.parse_args()

# load model
model_path = args.model
model = read_jdc_model(model_path)
print(f'Model loaded from {model_path}')
print(model)

# determine output files' paths
out_dir = args.out_dir
name = args.name
audio_path = args.input_audio

# path to save raw melody sources
melody_raw_path = os.path.join(out_dir, f'{name}.pkl')
# path to save the actual midi file
midi_out_file = os.path.join(out_dir, f'{name}.mid')
print(f'Melody raw data path: {melody_raw_path}')
print(f'MIDI output file: {midi_out_file}')

# extract melody and save resulting MIDI files
melody, timestamp = extract_melody(
    audio_path, model, PitchLabeler(), save_to=melody_raw_path)
save_midi(melody, timestamp, output_file=midi_out_file)
