from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional
import numpy as np

def plot_sequence(sequence:NDArray, dt:float, title:Optional[str] = 'unknown')-> None:
    N, n = sequence.shape
    t = np.linspace(0,(N-1)*dt,N)
    fig, axs = plt.subplots(figsize=(10,5),nrows=1,ncols=n, tight_layout=True, squeeze=0)
    for state_idx, (state_sequence, ax) in enumerate(zip(sequence.T, axs.reshape(-1))):
        ax.plot(t, state_sequence)
        ax.set_xlabel(f'time, dt = {dt:.2f}')
        ax.set_title(f'{title} state {state_idx}')
        ax.grid()
    

