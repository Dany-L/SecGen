from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List
import numpy as np
import os

def plot_sequence(sequence:List[NDArray], dt:float, title:Optional[str] = 'unknown', legend:Optional[List[str]]=None)-> Figure:
    N, n = sequence[0].shape
    t = np.linspace(0,(N-1)*dt,N)
    if legend is None:
        legend = ['unknown' for n in range(n)]
    fig, axs = plt.subplots(figsize=(10,5),nrows=1,ncols=n, tight_layout=True, squeeze=0)
    for state_idx, ax in enumerate(axs.reshape(-1)):
        for seq, label in zip(sequence, legend):
            ax.plot(t, seq[:,state_idx], label=label)
        ax.set_xlabel(f'time, dt = {dt:.2f}')
        ax.set_title(f'{title} state {state_idx}')
        ax.grid()
        ax.legend()
    return fig

def save_fig(fig:Figure, name:str, directory: str)->None:
    fig.savefig(os.path.join(directory,f'{name}.jpg'))
    plt.close(fig)

