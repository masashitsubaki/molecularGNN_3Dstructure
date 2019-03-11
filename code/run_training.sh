#!/bin/bash

DATASET='QM9_molecularsize<=15'
# DATASET='QM9_full'

# property='HOMO(eV)'
# property='LUMO(eV)'
property='U0(eV)'

update=sum
# update=mean

output=sum
# output=mean

dim=25
layer_hidden=6
layer_output=3
batch=16
lr=1e-3
lr_decay=0.95
decay_interval=10
weight_decay=1e-6
iteration=1000

setting=$DATASET--$property--$update--$output--dim$dim--layer_hidden$layer_hidden--layer_output$layer_output--batch$batch--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--weight_decay$weight_decay--iteration$iteration
python run_training.py $DATASET $property $update $output $dim $layer_hidden $layer_output $batch $lr $lr_decay $decay_interval $weight_decay $iteration $setting
