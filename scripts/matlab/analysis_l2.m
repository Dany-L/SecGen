clear all, close all,
%%

experiment_name = 'MSD-16';

result_directory = '~/coupled-msd/2025_01_25-cRnn';
test_file_name = '~/coupled-msd/data/coupled-msd-routine/processed/test/0093_simulation_T_1500.csv';

% result_directory = '~/actuated_pendulum/results_local';
% test_file_name = '~/actuated_pendulum/data/nonlinear-initial_state-0_M-500_T-10/processed/test/0198_simulation_T_10.csv';


% model_names = {'tanh','dzn','dznGen'};
model_names = {'satGen'};
results = cell(length(model_names));
results_wc =cell(length(model_names));
for model_idx =1:length(model_names)
    model_name = model_names{model_idx};
    fprintf('---%s---\n', model_name)

    e_m_name = sprintf('%s-%s', experiment_name, model_name);
    parameter_file_name = sprintf('model_params-%s.mat', e_m_name);
    
%     test_file_name = '/Users/jack/actuated_pendulum/data/ood-initial_state_0-s_4_M-100_T-10/processed/test/0058_simulation_T_10.csv';
    experiment_config_file_name = sprintf('config-experiment-%s.json', e_m_name);
    model_config_file_name = sprintf('config-model-%s.json', e_m_name);
    model_cfg = jsondecode(fileread(fullfile(result_directory,e_m_name,model_config_file_name)));
    experiment_cfg =jsondecode(fileread(fullfile(result_directory,e_m_name,experiment_config_file_name)));
    normalization = jsondecode(fileread(fullfile(result_directory,e_m_name,'normalization.json')));
    validation_log_file = fullfile(result_directory,e_m_name,'validation.log');

    switch model_cfg.nonlinearity
        case 'sat'
            varphi = @sat;
        case 'tanh'
            varphi = @tanh;
        case 'dzn'
            varphi = @dzn;
    end

    varphi_tilde = @(x) varphi(x) -x;
    % x = -5:0.1:5;
    % figure(), grid on, hold on
    % plot(x, varphi(x))
    % plot(x, varphi_tilde(x))


    h = experiment_cfg.horizons.testing;dt = experiment_cfg.dt; w=experiment_cfg.window;
    tab = readtable(test_file_name);
    d = tab.(experiment_cfg.input_names{:});nd =length(experiment_cfg.input_names);
    d = d(w+1:w+h,:);
    d_n = (d-normalization.input_mean)./normalization.input_std;
    e = tab.(experiment_cfg.output_names{:});ne = length(experiment_cfg.output_names);
    e = e(w+1:w+h,:);
    t = linspace(0, (h-1)*dt, h);

    a=-1;b=0;   

    load(fullfile(result_directory,e_m_name, parameter_file_name))

    if not(exist('H', 'var'))
        H = false;
    end

    %% load controller parameters
    nx = size(A_tilde,2); nd = size(B_tilde,2); nw = size(B2_tilde,2); 
    ne = size(C,1); nz = nw;
    % nx = size(A,1); nw=size(B2,1); nz=nw; ne=size(C,1);
    fprintf('Multiplier type: %s\n', model_cfg.multiplier)
    switch model_cfg.multiplier
        case 'none'
            L = eye(nw);
        case 'diag'
            L = diag(L);
    end
    
    P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];
    P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

    % ga2 = 0.001;
    ga2 = model_cfg.ga2;
    X = Lx * Lx';
    M11_orig = [-X,zeros(nx,nd), -C2_tilde';
        zeros(nd,nx), -ga2*eye(nd), -D21_tilde';
        -C2_tilde, -D21_tilde, -2*L];
    M21_orig = [A_tilde, B_tilde, B2_tilde;
        C, D, D12];
    M22_orig = [-X, zeros(nx,ne);
        zeros(ne,nx), -eye(ne)];
    M_orig = [M11_orig, M21_orig';M21_orig, M22_orig];

    if not(H==false)
        M_gen = [-eye(nz), H';H, -X];
        fprintf('max real eig M_orig: %f, max real eig M_gen: %f\n',max(real(eig(M_orig))), max(real(eig(M_gen))))
    else
        fprintf('max real eig M_orig: %f\n',max(real(eig(M_orig))))
    end

    X_inv = X^(-1);
    A = X_inv * A_tilde;
    B = X_inv * B_tilde;
    B2 = X_inv * B2_tilde;

    L_inv = L^(-1);
    C2 = L_inv * C2_tilde + H;
    D21 = L_inv * D21_tilde;

    sys = struct('A', A, 'B', B, 'B2', B2, 'C', C, 'D', D, 'D12', D12, 'C2', C2, 'D21', D21);
%     sys = struct('A', A, 'B', zeros(nx,nd), 'B2', B2, 'C', zeros(ne,nx), 'D', zeros(ne,nd), 'D12', zeros(ne,nw), 'C2', C2, 'D21', zeros(nz,nd))
    A_bar = (A-B2*C2);
    B_bar = (B-B2*D21);
    C_bar = (C-D12*C2);
    D_bar = (D-D12*D21);
    sys_tilde = struct('A', A_bar, 'B', B_bar, 'B2', B2, 'C', C_bar, 'D', D_bar, 'D12', D12, 'C2', C2, 'D21', D21);

    %% find an upper bound on the l2 gain

    fprintf('gen sector conditions\n')
    analyze_system(sys,-1,1,H);
    fprintf('std sector conditions\n')
    analyze_system(sys_tilde,0,1,false);
    
    % write l2 gain to validation log
%     fid = fopen(validation_log_file,'a+');
%     fprintf(fid,'l2 gain: %f\n',sqrt(double(ga2)));
%     fclose(fid);

            

    %% simulate
    e_hat_n_cmp = d_sim(sys, d_n, zeros(nx,1), varphi_tilde);
    e_hat_n = d_sim(sys_tilde, d_n, zeros(nx,1), varphi);
    assert(norm(e_hat_n - e_hat_n_cmp) < 1e-5)
    e_hat = e_hat_n .* normalization.output_std + normalization.output_mean;
    results{model_idx} = e_hat;

    %% simulate worst case amplification from lstm
    wc_lstm_filename = '/Users/jack/coupled-msd/2024_12_12-cRnn/MSD-128-zero-dual-lstm/seq/test_output-stability_l2-coupled-msd-routine.mat';
    wc_lstm = load(wc_lstm_filename);
    d_n_wc = squeeze(wc_lstm.d);
    % e_hat_n_wc = d_sim(sys, d_n_wc, zeros(nx,1), varphi);
    e_hat_n_wc = d_sim(sys_tilde, d_n_wc, zeros(nx,1), varphi_tilde);
    fprintf('%s: ga= %f (lstm: ga= %f) \n',model_name, sqrt(norm(e_hat_n_wc)^2/norm(d_n_wc)^2), sqrt(norm(squeeze(wc_lstm.e_hat))^2/norm(d_n_wc)^2))
    e_hat_wc = e_hat_n_wc .* normalization.output_std + normalization.output_mean;
    results_wc{model_idx} = {e_hat_wc, squeeze(wc_lstm.e_hat)};

end
figure(), grid on, hold on
for i =1:length(results)
    plot(t,results{i}')
end
plot(t,e', '--')
plot(t,d', '--')
legend([model_names, 'e', 'd'])


% for i =1:length(results_wc)
%     figure(), grid on, hold on
%     plot(results_wc{i}{1}')
%     plot(results_wc{i}{2}')
%     legend({model_names{i}, 'lstm'})
%     title(sprintf('%s: Worst case amplification from LSTM', model_names{i}))
% end






