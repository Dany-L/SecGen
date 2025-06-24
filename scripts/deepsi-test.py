import numpy as np
import deepSI as dsi
import nonlinear_benchmarks
import torch
from matplotlib import pyplot as plt

nx, nb, na = 2, 5, 5
np.random.seed(0)
torch.manual_seed(1)

def run_training():

    train_val, test = nonlinear_benchmarks.F16(atleast_2d=True)
    train, val = train_val[:4], train_val[4:]

    nu, ny, norm = dsi.get_nu_ny_and_auto_norm(train) 
    # Equivilent to 
    # nu = 'scalar', ny = 'scalar', norm = dsi.Norm(umean=train.u.mean(0), ustd=train.u.std(0), ymean=train.y.mean(0), ystd=train.y.std(0))
    
    model = dsi.SUBNET(nu, ny, norm, nx=nx, nb=nb, na=na)
    # this model has three components:
    # print(model.f)
    # print(model.h)
    # print(model.encoder)

    x = torch.randn(1,model.nx) # 1 batch size and nx state vector size 
    u = torch.randn(1) #1 batch size and 'scalar' input size
    print(f'{model.f(x, u) = }') #the components can be evaluated in desired point
    print(f'u: {u.shape}, x: {x.shape}')

    train_dict = dsi.fit(model, train=train, val=val, n_its=500, T=30, batch_size=64, val_freq=100)

    plt.figure(figsize=(5,2))
    test_sim = model.simulate(test)
    plt.plot(test_sim.y) #simulate is apply_experiment equivilent
    plt.plot(test.y)
    plt.show()
    from nonlinear_benchmarks.error_metrics import NRMSE, RMSE
    print(f'NRMS={NRMSE(test.y[model.na:], test_sim.y[model.na:]):.2%}')
    print(f'RMSE={RMSE(test.y[model.na:], test_sim.y[model.na:]):.4f} V')

if __name__ == "__main__":
    run_training()