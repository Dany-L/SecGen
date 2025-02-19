from nfoursid.nfoursid import NFourSID
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from crnn.data_io import load_data
from crnn.utils import base as utils

dataset_dir = os.path.expanduser('~/coupled-msd/data/coupled-msd-routine/processed')
input_names = ['u_1']
output_names = ['y_1']

# load training data
train_inputs, train_outputs = load_data(
    input_names,
    output_names,
    "train",
    dataset_dir,
)
input_mean, input_std = utils.get_mean_std(train_inputs)
output_mean, output_std = utils.get_mean_std(train_outputs)
n_train_inputs = utils.normalize(train_inputs, input_mean, input_std)
n_train_outputs = utils.normalize(train_outputs, output_mean, output_std)

# %%
print(len(n_train_inputs))
print(n_train_inputs[0].shape)

io_data = pd.DataFrame(
    np.hstack([np.vstack(n_train_inputs), np.vstack(n_train_outputs)]),
    columns=input_names + output_names
)

n4sid = NFourSID(io_data, input_columns=input_names, output_columns=output_names)
n4sid.subspace_identification()
nx = 8
ss, cov = n4sid.system_identification(rank=nx)




