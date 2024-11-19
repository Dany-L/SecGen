clear all, close all,
%%
experiment_name = 'P-16-zero';
result_directory = '~/cloud_privat/03_Promotion/_transfer';

% model_names = {'tanh','dzn','dznGen'};
model_names = {'l2-tanh'};
results = cell(length(model_names));
for model_idx =1:length(model_names)
    model_name = model_names{model_idx};
    fprintf('---%s---\n', model_name)

    e_m_name = sprintf('%s-%s', experiment_name, model_name);
    parameter_file_name = sprintf('model_params-%s.mat', e_m_name);
    test_file_name = '/Users/jack/actuated_pendulum/data/nonlinear-initial_state-0_M-500_T-10/processed/test/0198_simulation_T_10.csv';
%     test_file_name = '/Users/jack/actuated_pendulum/data/ood-initial_state_0-s_4_M-100_T-10/processed/test/0058_simulation_T_10.csv';
    experiment_config_file_name = sprintf('config-experiment-%s.json', e_m_name);
    model_config_file_name = sprintf('config-model-%s.json', e_m_name);
    model_cfg = jsondecode(fileread(fullfile(result_directory,e_m_name,model_config_file_name)));
    experiment_cfg =jsondecode(fileread(fullfile(result_directory,e_m_name,experiment_config_file_name)));
    normalization = jsondecode(fileread(fullfile(result_directory,e_m_name,'normalization.json')));
    validation_log_file = fullfile(result_directory,e_m_name,'validation.log');

    h = experiment_cfg.horizons.testing;dt = experiment_cfg.dt; w=experiment_cfg.window;
    tab = readtable(test_file_name);
    d = tab.(experiment_cfg.input_names{:});nd =length(experiment_cfg.input_names);
    d = d(w+1:w+h,:);
    d_n = (d-normalization.input_mean)./normalization.input_std;
    e = tab.(experiment_cfg.output_names{:});ne = length(experiment_cfg.output_names);
    e = e(w+1:w+h,:);
    t = linspace(0, (h-1)*dt, h);

    a=0;b=1;   

    load(fullfile(result_directory,e_m_name, parameter_file_name))

    if exist('H', 'var')
        b_gen=true;
    else
        b_gen=false;
    end

    %% load controller parameters
%     nx = size(A_tilde,1); nw = size(B2_tilde,1); nz = nw;
    nx = size(A,1); nw=size(B2,1); nz=nw; ne=size(C,1);
    fprintf('Multiplier type: %s\n', model_cfg.multiplier)
    switch model_cfg.multiplier
        case 'none'
            L = eye(nw);
        case 'diag'
            L = diag(L);
    end
    
    P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];
    P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

%             M11 = trans.torch_bmat(
%                 [[-self.ga2, self.D21_tilde.T], [self.D21_tilde, -2 * L]]
%             )
%             M21 = trans.torch_bmat([[self.D, self.D12]])

    M11_orig = [-ga2, D21_tilde';D21_tilde, -2*L ];
    M21_orig = [D, D12];
    M_orig = [M11_orig, M21_orig';M21_orig, -eye(ne)];
    fprintf('max real eig M_orig: %f\n',max(real(eig(M_orig))))

    D21 = L^(-1) * D21_tilde;
    sys = struct('A', A, 'B', B, 'B2', B2, 'C', C, 'D', D, 'D12', D12, 'C2', C2, 'D21', D21);
%     sys = struct('A', A, 'B', zeros(nx,nd), 'B2', B2, 'C', zeros(ne,nx), 'D', zeros(ne,nd), 'D12', zeros(ne,nw), 'C2', C2, 'D21', zeros(nz,nd))


    %% find an upper bound on the l2 gain
    eps=1e-5;

    X = sdpvar(nx,nx);
    lambda = sdpvar(nw,1);
    ga2 = sdpvar(1,1);
    
    L1 = [eye(nx),zeros(nx,nd), zeros(nx,nw);
        A, B, B2];
    L2 = [zeros(nd,nx), eye(nd), zeros(nd,nw);
        C, D, D12];

    if b_gen
        L3 = [zeros(nw,nx), zeros(nw,nd), eye(nw);
            C2-H, D21, zeros(nz,nw)];
        add_constr = ([-X, H';H, -eye(nx)]<= -eps*eye(2*nx));
    else
        L3 = [zeros(nw,nx),zeros(nw,nd) eye(nw);
            C2, D21, zeros(nz,nw)];
        add_constr = [];
    end

    P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];
    L = diag(lambda); % can be replaced by diag multiplier
    P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

    lmis = [];
    lmi = L1' * [-X, zeros(nx,nx); zeros(nx,nx), X] * L1 + ...
        L2' * [-ga2*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)] * L2 + ...
        L3' * P * L3;

    lmis = lmis + (lmi <= -eps * eye(size(L1,2)));
    lmis = lmis + (X >= eps*eye(nx));
    lmis = lmis + add_constr;
     
    sol = optimize(lmis, [], sdpsettings('solver','MOSEK','verbose', 0));
    if sol.problem == 0
        fprintf('parameters have optimal gamma: %g \n', sqrt(double(ga2)))
    else
        fprintf('Trained parameters are not feasible: %s \n', sol.info)
    end

    % write l2 gain to validation log
%     fid = fopen(validation_log_file,'a+');
%     fprintf(fid,'l2 gain: %f\n',sqrt(double(ga2)));
%     fclose(fid);

            

    %% simulate
    e_hat_n = d_sim(sys, d_n, zeros(nx,1), @dzn);
    e_hat = e_hat_n .* normalization.output_std + normalization.output_mean;
    results{model_idx} = e_hat;

end
figure(), grid on, hold on
for i =1:length(results)
    plot(t,results{i}')
end
plot(t,e', '--')
plot(t,d', '--')
legend([model_names, 'e', 'd'])





