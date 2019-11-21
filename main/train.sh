#!/bin/bash

# The dataset.
dataset=QM9_under14atoms
# dataset=yourdataset

# The molecular property to be learned.
property='U0(kcalmol^-1)'
# property='HOMO(eV)'
# property='LUMO(eV)'

# The setting of a neural network architecture.
dim=200
layer_hidden=6
layer_output=6

# The setting for optimization.
batch_train=32
batch_test=32
lr=1e-3
lr_decay=0.99
decay_interval=10
iteration=3000

setting=$dataset--$property--dim$dim--layer_hidden$layer_hidden--layer_output$layer_output--batch_train$batch_train--batch_test$batch_test--lr$lr--lr_decay$lr_decay--decay_interval$decay_interval--iteration$iteration
python train.py $dataset $property $dim $layer_hidden $layer_output $batch_train $batch_test $lr $lr_decay $decay_interval $iteration $setting
