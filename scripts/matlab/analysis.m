clear all, close all,
%%
model_name = 'dznGen';
result_directory = '~/actuated_pendulum/results_local/';
parameter_file_name = sprintf('model_params-%s.mat', model_name);
test_file_name = '/Users/jack/actuated_pendulum/nonlinear-initial_state-0_M-500_T-10/processed/test/0281_simulation_T_10.csv';
config_file_name = sprintf('config-%s.json', model_name);
cfg = jsondecode(fileread(fullfile(result_directory,model_name,config_file_name)));

h = cfg.horizons.testing;dt = cfg.dt; w=cfg.window;
tab = readtable(test_file_name);
d = tab.(cfg.input_names{:});nd =length(cfg.input_names);
d = d(w+1:w+h,:);
e = tab.(cfg.output_names{:});ne = length(cfg.output_names);
e = e(w+1:w+h,:);
t = linspace(0, (h-1)*dt, h);

load(fullfile(result_directory,model_name,parameter_file_name))

%%
nx = size(A_tilde,1); nw = size(B2_tilde,1); nz = nw;
M11_orig = [-X, C2';C2, -2*eye(nz) ];
M21_orig = [A_tilde, B2_tilde];
M_orig = [M11_orig, M21_orig';M21_orig, -X];
fprintf('max real eig M_orig: %f\n',max(real(eig(M_orig))))

X_inv = X^(-1);
A = X_inv * A_tilde;
B2 = X_inv * B2_tilde;


L1 = [eye(nx), zeros(nx,nw);
    A, B2];
L2 = [zeros(nw,nx), eye(nw);
    C2-H, zeros(nz,nw)];
% L2 = [zeros(nw,nx), eye(nw);
%     C2, zeros(nz,nw)];

M = L1' * [-X, zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
    L2' * [-2*eye(nw), eye(nw); eye(nw), zeros(nw,nw)] * L2;
fprintf('max real eig M: %f\n',max(real(eig(M))))
fprintf('max real eig (H^T H - X): %f\n',max(real(eig(H'*H-X))))
%%
x = zeros(h+1,nx);w = zeros(h,nw);e_hat = zeros(h,ne);z =zeros(h,nz);
x(1,:) = zeros(1,nx); % initial condition
for k=1:h
    z(k,:) = C2 * x(k,:)' + D21 * d(k,:)';
    w(k,:) = dzn(z(k,:));
    x(k+1,:) = A * x(k,:)' + B * d(k,:)' + B2 * w(k,:)';
    e_hat(k,:)= C * x(k,:)' + D * d(k,:)' + D12 * w(k,:)';
end
figure(), grid on, hold on
plot(t,e_hat')
plot(t,e')
legend('e hat', 'e')





