clear all, close all
%%
syms c2 d21 d12 d11
syms x0 d0 e0

x1 = tanh(c2*x0 + d21*d0);
e_hat0 = d12*x1 + d11*d0;

l = (e_hat0-e0)^2;

e0 = 1;x0=1;d0=1;

G = gradient(subs(l))

c2=0;d21=0;d12=0;
subs(G)

%% 
syms C2 [2 2]
syms D21 [2 1]
syms D12 [1 2]
syms x0 [2 1]
syms d1 d0 e1 e0 x

x1 = tanh(C2*x0 + D21*d0);
e_hat0 = D12*x1;
x2 = tanh(C2*x1 + D21*d1);
e_hat1 = D12*x2;


l = 1/2*((e_hat0 - e0)^2 +(e_hat1 - e1)^2);

x01 = 1; x02=1;
d0=0.2; d1=0.3;
e0=1; e1 = 1;
% gradient(subs(e_hat0))
% simplify(gradient(subs(e_hat1)))
G = gradient(subs(l));
length(G)

% matlabFunction(subs(l))

% C21_1 = -0.2069; C21_2=  1.0523;
% C22_1 = -0.2212; C22_2 =  -2.9653;
% D211 = -0.4844;
% D212 =-0.1699;
% D121 = -0.9700; D122= -0.4767;

C21_1 = 0; C21_2=  0;
C22_1 = 0; C22_2 =  0;
D211 = 0;
D212 =0;
D121 = 0; D122= 0;


double(subs(l))
double(subs(G))
double(subs(e_hat0))
double(subs(e_hat1))



