clear all, close all,
%%
model_name = 'dznGen';
result_directory = '~/actuated_pendulum/results_local/';
parameter_file_name = sprintf('model_params-%s.mat', model_name);

load(fullfile(result_directory,model_name,parameter_file_name))

%%
X_inv = X^(-1);
A = X_inv * A_tilde;
B2 = X_inv * B2_tilde;
nx = size(A,1); nw = size(B2,1); nz = nw;

L1 = [eye(nx), zeros(nx,nw);
    A, B2];
L2 = [zeros(nw,nx), eye(nw);
    C2-H, zeros(nz,nw)];

M = L1' * [-X, zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
    L2' * [-2*eye(nw), eye(nw); eye(nw), zeros(nw,nw)] * L2;
fprintf('max real eig M: %f\n',max(real(eig(M))))
fprintf('max real eig (H^T H - X): %f\n',max(real(eig(H'*H-X))))


