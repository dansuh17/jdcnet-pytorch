#!/usr/bin/env python3
"""
Preprocessing script for MedleyDB Melody Subset.
"""
import argparse
from jdc.preprocess import preprocess

parser = argparse.ArgumentParser(
    description='MedleyDB melody subset preprocessing script')
parser.add_argument('-i', '--in_root', type=str,
                    help='root directory of medleyDB dataset melody subset')
parser.add_argument('-o', '--out_root', type=str,
                    help='data output root directory')
parser.add_argument('-m', '--metadata_path', type=str,
                    default='MedleyDB-Melody/medleydb_melody_metadata.json',
                    help='path to MedleyDB metadata file (JSON)')
args = parser.parse_args()

in_root = args.in_root
out_root = args.out_root
# metadata JSON file included in MedleyDB dataset
metadata_path = args.metadata_path

# preprocess the entire dataset to `out_root` into trainable format
preprocess(in_root, out_root, metadata_path)
