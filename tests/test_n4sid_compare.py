import os
import numpy as np
import scipy.io
import pandas as pd
import pytest
import matplotlib.pyplot as plt
import json
from utils import load_filtered_csvs

from crnn.models.base import N4SID, N4SID_NG_with_nfoursid

def compute_fit_percent(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    # Returns fit percentage for each output channel
    fit = 100 * (1 - np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true - y_true.mean(axis=1, keepdims=True), axis=1))
    return fit

def simulate_linear_system(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    u: np.ndarray,
    x0: np.ndarray = None
) -> np.ndarray:
    """
    Simulate output of a discrete-time linear system:
    x_{t+1} = A x_t + B u_t
    y_t = C x_t + D u_t
    u: shape [n_inputs, T]
    Returns y: shape [n_outputs, T]
    """
    nx = A.shape[0]
    n_outputs = C.shape[0]
    T = u.shape[1]
    x = np.zeros((nx, T))
    y = np.zeros((n_outputs, T))
    if x0 is None:
        x_prev = np.zeros((nx, 1))
    else:
        x_prev = x0.reshape(nx, 1)
    for t in range(T):
        u_t = u[:, t:t+1]
        x[:, t:t+1] = A @ x_prev + B @ u_t
        y[:, t:t+1] = C @ x[:, t:t+1] + D @ u_t
        x_prev = x[:, t:t+1]
    return y

def compare_eigenvalues(A_py: np.ndarray, A_mat: np.ndarray):
    eig_py = np.linalg.eigvals(A_py)
    eig_mat = np.linalg.eigvals(A_mat)
    print("Python A eigenvalues:", eig_py)
    print("MATLAB A eigenvalues:", eig_mat)
    print("Max abs eigenvalue (Python):", np.max(np.abs(eig_py)))
    print("Max abs eigenvalue (MATLAB):", np.max(np.abs(eig_mat)))

@pytest.mark.parametrize("data_directory", [os.path.expanduser("~/F16/data/F16GVT_Files/BenchmarkData")])
@pytest.mark.parametrize("input_names", [['Force']])
@pytest.mark.parametrize("output_names", [['Acceleration1', 'Acceleration2', 'Acceleration3']])
def test_n4sid_vs_matlab(
    data_directory: str,
    input_names: list[str],
    output_names: list[str],
    tmp_path,
    mat_file: str = "F16_none_n4sid.mat",
    nx: int = 8,
    atol: float = 1e-2,
    rtol: float = 1e-2
):
    # Load MATLAB results
    tmp_path.mkdir(parents=True, exist_ok=True)

    root_path = os.path.dirname(__file__)
    mat_path = os.path.join(root_path, "data", "n4sid", mat_file)
    mat = scipy.io.loadmat(mat_path)
    sys_struct = mat["sys_struct"][0, 0]
    A_mat = sys_struct["A"]
    B_mat = sys_struct["B"]
    C_mat = sys_struct["C"]
    D_mat = sys_struct["D"]
    fit_percent_matlab = np.squeeze(sys_struct["n4sid_info"]["Fit"][0, 0]["FitPercent"][0,0])

    positive_filter = []
    exclude_filter = ['SpecialOdd', 'Validation']
    ds, es = load_filtered_csvs(data_directory, positive_filter, exclude_filter, input_names, output_names)

    ds_mean, ds_std = ds.mean(axis=1, keepdims=True), ds.std(axis=1, keepdims=True)
    es_mean, es_std = es.mean(axis=1, keepdims=True), es.std(axis=1, keepdims=True)
    ds_norm = (ds - ds_mean) / ds_std
    es_norm = (es - es_mean) / es_std

    # Run Python N4SID
    # A_py, B_py, C_py, D_py, _, _ = N4SID_NG_with_nfoursid(ds_norm, es_norm, nx, enforce_stability_method="projection")
    A_py, B_py, C_py, D_py, _, _ = N4SID(ds_norm, es_norm, nx,require_stable=True, enforce_stability_method="projection")

    # Compare eigenvalues
    compare_eigenvalues(A_py, A_mat)

    # Simulate outputs for both models
    y_pred_py = simulate_linear_system(A_py, B_py, C_py, D_py, ds_norm)
    y_pred_mat = simulate_linear_system(A_mat, B_mat, C_mat, D_mat, ds_norm)

    fit_percent_py = compute_fit_percent(es_norm, y_pred_py)
    fit_percent_mat = compute_fit_percent(es_norm, y_pred_mat)

    print(f'Python FitPercent: {fit_percent_py}, Simulated Matlab FitPercent: {fit_percent_mat}, Original Matlab FitPercent: {fit_percent_matlab}')

    # Save fit percentages to a JSON file
    fit_results = {
        "MATLAB_FitPercent": fit_percent_matlab.tolist(),
        "Python_FitPercent": fit_percent_py.tolist(),
        "MATLAB_Simulated_FitPercent": fit_percent_mat.tolist()
    }
    with open(tmp_path / "fit_results.json", "w") as f:
        json.dump(fit_results, f, indent=4)

    # Save plots for each output channel
    for i in range(es_norm.shape[0]):
        plt.figure()
        plt.plot(es_norm[i], label="True")
        plt.plot(y_pred_py[i], label="Python N4SID")
        plt.plot(y_pred_mat[i], label="MATLAB N4SID")
        plt.title(f"Output channel {i} - Fit: Python={fit_percent_py[i]:.2f}%, MATLAB={fit_percent_mat[i]:.2f}%")
        plt.legend()
        plot_path = tmp_path / f"output_channel_{i}.png"
        plt.savefig(plot_path)
        plt.close()

    # Assertions for fit percentages
    for i, (fit_py, fit_mat) in enumerate(zip(fit_percent_py, fit_percent_mat)):
        assert abs(fit_py - fit_mat) < 30, f"FitPercent mismatch for output {i}: Python={fit_py:.2f}, MATLAB={fit_mat:.2f}"


