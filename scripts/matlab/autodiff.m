syms c2 d21 d12 x0 d0 d1

x1 = tanh(c2*x0 + d21*d0);
e_hat_0 = d12*x1;

diff(e_hat_0, d12)
diff(e_hat_0, c2)
diff(e_hat_0, d21)

x2 = tanh(c2*x1+d21*d1);
e_hat_1 = d12*x2;

diff(e_hat_1, d12)
diff(e_hat_1, c2)
diff(e_hat_1, d21)
