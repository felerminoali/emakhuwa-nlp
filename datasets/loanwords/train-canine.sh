#!/bin/bash

echo "Running Python code CANINE ONE"
model_file="checkpoints/model_one.pt"
python train-canine-one.py -ms "$model_file" -psp predictions/canine/ -dp data/canine/ -mn google/canine-c -tn google/canine-c

echo "Running Python code CANINE CONCAT"
model_file="checkpoints/model_concat.pt"
python train-canine-concat.py -ms "$model_file" -psp predictions/canine/ -dp data/${pair}/canine/ -mn google/canine-c -tn google/canine-c
