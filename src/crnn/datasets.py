from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset


class RecurrentWindowHorizonDataset(Dataset):
    def __init__(
        self,
        input_seqs: List[NDArray[np.float64]],
        output_seqs: List[NDArray[np.float64]],
        horizon: int,
        window: int,
    ):
        self.h = horizon
        self.w = window
        self.nd = input_seqs[0].shape[1]
        self.ne = output_seqs[0].shape[1]

        (
            self.d_init,
            self.e_init,
            self.d,
            self.e,
        ) = self.__load_data(input_seqs, output_seqs)

    def __load_data(
        self,
        d_seqs: List[NDArray[np.float64]],
        e_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        d_init_seq = list()
        e_init_seq = list()
        d_seq = list()
        e_seq = list()

        for ds, es in zip(d_seqs, e_seqs):
            n_samples = int(ds.shape[0] / (self.w + self.h + 1))

            d_init = np.zeros((n_samples, self.w, self.nd), dtype=np.float64)
            e_init = np.zeros((n_samples, self.w, self.ne), dtype=np.float64)

            d = np.zeros((n_samples, self.h, self.nd), dtype=np.float64)
            e = np.zeros((n_samples, self.h, self.ne), dtype=np.float64)

            for idx in range(n_samples):
                time = idx * self.h

                # inputs
                d_init[idx, :, : self.nd] = ds[time + 1 : time + self.w + 1, :]
                d[idx, :, :] = ds[time + self.w + 1 : time + self.h + self.w + 1, :]
                # outputs
                e_init[idx, :] = es[time + 1 : time + self.w + 1, :]
                e[idx, :, :] = es[time + self.w + 1 : time + self.h + self.w + 1, :]

            d_init_seq.append(d_init)
            e_init_seq.append(e_init)
            d_seq.append(d)
            e_seq.append(e)

        return (
            np.vstack(d_init_seq),
            np.vstack(e_init_seq),
            np.vstack(d_seq),
            np.vstack(e_seq),
        )

    def __len__(self) -> int:
        return self.d_init.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            "d_init": self.d_init[idx],
            "e_init": self.e_init[idx],
            "d": self.d[idx],
            "e": self.e[idx],
        }
