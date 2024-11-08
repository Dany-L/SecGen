from crnn.train import train
from crnn.evaluate import evaluate
import sys
import os
import torch

torch.set_default_dtype(torch.double)

root_dir = os.path.split(os.path.abspath(sys.argv[0]))[0]
print(root_dir)
result_directory = os.path.join(root_dir,'_tmp/results')
configuration_file = os.path.join(root_dir,'config.json')
data_directory = os.path.join(root_dir,'data')

with open(os.path.join(root_dir,'models.txt')) as f:
    model_names = [model_name.strip() for model_name in f.readlines()]
with open(os.path.join(root_dir,'experiments.txt')) as f:
    experiment_names = [experiment_name.strip() for experiment_name in f.readlines()]

for model_name in model_names:
    for experiment_name in experiment_names:
        print(f'Experiment: {experiment_name}, Model: {model_name}')
        train(configuration_file,data_directory,result_directory, model_name, experiment_name)
        evaluate(configuration_file,data_directory,result_directory, model_name, experiment_name)
