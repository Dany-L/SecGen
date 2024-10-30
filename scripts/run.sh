#!/bin/bash

export DATASET_DIRECTORY=~/actuated_pendulum/nonlinear-initial_state-0_M-500_T-10/processed
export RESULT_DIRECTORY=~/actuated_pendulum/results_local
export CONFIG_DIRECTORY=~/Documents/01_Git/01_promotion/crnn/config

while read -r line;
do echo "${line}"
    python ~/Documents/01_Git/01_promotion/crnn/scripts/train_model.py --model_name "${line}"
    python ~/Documents/01_Git/01_promotion/crnn/scripts/evaluate_model.py --model_name "${line}"
done < $1

