#!/bin/bash

DATASET='QM9_molecularsize<=15'
# DATASET='QM9_full'

python preprocess_data.py $DATASET
