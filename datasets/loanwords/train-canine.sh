#!/bin/bash

echo "Running Python code CANINE ONE"
python train-canine-one.py -ms "$model_file" -psp predictions/canine/ -dp data/canine/ -mn google/canine-c -tn google/canine-c

echo "Running Python code CANINE CONCAT"
python train-canine-concat.py -ms "$model_file" -psp predictions/canine/ -dp data/canine/ -mn google/canine-c -tn google/canine-c