% we want to test if stability implies finite gain l2-stability for
% sufficently large gamma

nx = 2; nw=1;nz=nw;nd=1;ne=1;
a = 0; b = 1;

%% first assume no multiplier
A_tilde = sdpvar(nx,nx);
B2_tilde = sdpvar(nx,nw);

C2 = sdpvar(nz,nx);

X = sdpvar(nx,nx);

P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];
L = eye(nw); % can be replaced by diag multiplier
P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

M11 = [-X, C2'; C2, -2*L];
M21 = [A_tilde,B2_tilde];
M = [M11,M21';M21,-X];
 
sol = optimize(M<=-eps*eye(size(M,1)), [], sdpsettings('solver','MOSEK','verbose', 0));

X = double(X);
Xinv = X^(-1);
A = Xinv * double(A_tilde);
B2 = Xinv * double(B2_tilde);
C2 = double(C2);

L1 = [eye(nx), zeros(nx,nw);
    A, B2];
L3 = [zeros(nz,nx), eye(nw);
    C2, zeros(nz,nw)];
M = L1' * [-X, zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
    L3' * P * L3;
fprintf('max real eig M: %f\n', max(real(eig(M))))
% use random matrices B, C, D, D12 and D21 and test if system is finite
% gain stable
B = rand(nx,nd);
C = rand(ne,nx); D = rand(ne,nd); D12=rand(ne,nw);
D21 = rand(nw,nd);

L1 = [eye(nx), zeros(nx,nd) zeros(nx,nw);
    A, B, B2];
L2 = [zeros(nd,nx), eye(nd) zeros(nd,nw);
    C, D, D12];
L3 = [zeros(nw,nx), zeros(nw,nd) eye(nw);
    C2, D21, zeros(nz,nw)];

M = @(ga) L1' * [-X, zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
    L2' * [-ga^2*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)]*L2 + ...
    L3' * P * L3;

gas = logspace(0,6,100);max_real_eig = zeros(length(gas),1);
for ga_idx = 1:length(gas)
    max_real_eig(ga_idx,1) = max(real(eig(M(gas(ga_idx)))));
end
figure(), semilogx(gas,max_real_eig), grid on


