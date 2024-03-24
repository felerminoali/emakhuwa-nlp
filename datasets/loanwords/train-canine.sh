#!/bin/bash

language_pairs=(
  "vmw-pt"
)

for pair in "${language_pairs[@]}"; do
  model_file="checkpoints/model_one_${pair}.pt"
  rm -r "$model_file"
  python train-canine-one.py -ms "$model_file" -psp predictions/canine/ -dp data/${pair}/canine/ -mn google/canine-c -tn google/canine-c
done

language_pairs=(
  "vmw-pt"
)

for pair in "${language_pairs[@]}"; do
  model_file="checkpoints/model_concat_${pair}.pt"
  rm -r "$model_file"
  python train-canine-concat.py -ms "$model_file" -psp predictions/canine/ -dp data/${pair}/canine/ -mn google/canine-c -tn google/canine-c
done