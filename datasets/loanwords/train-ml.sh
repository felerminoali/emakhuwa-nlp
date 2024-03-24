#!/bin/bash

# Run Python code RF
echo "Running Python code RF"
python "trainRF.py"

# Run Python code RL
echo "Running Python code RL"
python "trainRL.py"

# Run Python code SVM
echo "Running Python code SVM"
python "trainSVM.py"

# Run Python code NN
echo "Running Python code NN"
python "trainNN.py"