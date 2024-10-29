import argparse

import crnn.io as io_handling
import crnn.configuration as cfg
import crnn.utils.plot as plot
from crnn.datasets import RecurrentWindowHorizonDataset
from crnn.models import sector_bounded as models
from torch.utils.data import DataLoader
from typing import Any
import matplotlib.pyplot as plt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a CRNN model.')
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model to train')
    args = parser.parse_args()
    print(args)

    base_config = cfg.load_configuration()
    inputs, outputs = io_handling.load_data(base_config.input_names,base_config.output_names,'train')
    print(f'len input:{len(inputs)} shape input[0] {inputs[0].shape} len output:{len(outputs)} shape outputs[0] {outputs[0].shape}')
    # plot.plot_sequence(inputs[0], base_config.dt, 'input 0')
    # plot.plot_sequence(outputs[0], base_config.dt, 'output 0')
    # plt.show()
    rnn_dataset = RecurrentWindowHorizonDataset(inputs,outputs,base_config.horizons.training, base_config.window)
    data_loader = DataLoader(rnn_dataset, base_config.batch_size,drop_last=True)

    model = models.SectorBoundedLtiRnn(
        nz=base_config.nz,
        nd=len(inputs),
        ne=len(outputs),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=base_config.optimizer.learning_rate)


    


    # model = initialize_model(args.model_name)
    # data = load_data()
    # train_model(model, data)

if __name__ == '__main__':
    main()