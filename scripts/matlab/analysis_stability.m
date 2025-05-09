clear all, close all,
%%
experiment_names = {'MSD-32-joint','MSD-32-joint', 'MSD-32-joint'};
model_names = {'dzn','dznGen','tanh'};
% result_directory = '~/cloud_private/results_local/';
result_directory = '/Users/jack/coupled-msd/2024_11_21-cRnn';
% result_directory = '~/cloud_privat/03_Promotion/_transfer';

% model_names = {'tanh','dzn','dznGen'};
results = cell(length(model_names));
for model_idx =1:length(model_names)
    model_name = model_names{model_idx};
    experiment_name = experiment_names{model_idx};
    fprintf('---%s---\n', model_name)

    e_m_name = sprintf('%s-%s', experiment_name, model_name);
    parameter_file_name = sprintf('model_params-%s.mat', e_m_name);
%     test_file_name = '/Users/jack/actuated_pendulum/data/nonlinear-initial_state-0_M-500_T-10/processed/test/0198_simulation_T_10.csv';
%     test_file_name = '/Users/jack/actuated_pendulum/data/ood-initial_state_0-s_4_M-100_T-10/processed/test/0058_simulation_T_10.csv';
    test_file_name = '/Users/jack/coupled-msd/data/coupled-msd-routine/processed/test/0008_simulation_T_1500.csv';
    experiment_config_file_name = sprintf('config-experiment-%s.json', e_m_name);
    model_config_file_name = sprintf('config-model-%s.json', e_m_name);
    model_cfg = jsondecode(fileread(fullfile(result_directory,e_m_name,model_config_file_name)));
    experiment_cfg =jsondecode(fileread(fullfile(result_directory,e_m_name,experiment_config_file_name)));
    normalization = jsondecode(fileread(fullfile(result_directory,e_m_name,'normalization.json')));
    validation_log_file = fullfile(result_directory,e_m_name,'validation.log');
    evaluation_file_name = fullfile(result_directory, e_m_name, 'test-eval.json');

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
    nx = size(A_tilde,1); nw = size(B2_tilde,1); nz = nw;
    fprintf('Multiplier type: %s\n', model_cfg.multiplier)
    switch model_cfg.multiplier
        case 'none'
            L = eye(nw);
        case 'diag'
            L = diag(L);
    end
    
    P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];
    P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

    M11_orig = [-X, C2_tilde';C2_tilde, -2*L ];
    M21_orig = [A_tilde, B2_tilde];
    M_orig = [M11_orig, M21_orig';M21_orig, -X];
    fprintf('max real eig M_orig: %f\n',max(real(eig(M_orig))))

    X_inv = X^(-1);
    A = X_inv * A_tilde;
    B2 = X_inv * B2_tilde;

    C2 = L^(-1) * C2_tilde;
    sys = struct('A', A, 'B', B, 'B2', B2, 'C', C, 'D', D, 'D12', D12, 'C2', C2, 'D21', D21);
%     sys = struct('A', A, 'B', zeros(nx,nd), 'B2', B2, 'C', zeros(ne,nx), 'D', zeros(ne,nd), 'D12', zeros(ne,nw), 'C2', C2, 'D21', zeros(nz,nd));
    %% calculate rho and M
    % \|x^k\|<= m rho^k \|x^0\|
%     L1 = [eye(nx),zeros(nx,nw);
%         A, B2];
%     L2 = [zeros(nw,nx) eye(nw);
%             C2, zeros(nz,nw)];
% 
%     x = 0:0.1:10;max_ev=zeros(length(x),1);
%     M_fcn = @(x) L1' * [X*(-eye(nx)+x*eye(nx)), zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
%         L2' * P * L2;
%     ep_star = fsolve(@(x) max(real(eig(M_fcn(x)))), 0);
%     for x_idx = 1:length(x), max_ev(x_idx) = max(real(eig(M_fcn(x(x_idx))))); end
%     figure, plot(x,max_ev), grid on
%     fprintf('eps: %f, max eig M(eps): %f\n', ep_star, max(real(eig(M_fcn(ep_star)))))

    % simulate model for a few initial conditions and plot f(k) = m rho^k
    % \|x^0\|
%     h_stab=20;
%     low=-10;high=-low;
%     m_rho =@(k, x0) sqrt(max(real(eig(X)))/min(real(eig(X)))*(1-ep_star).^k)*norm(x0,2);
%     epochs = 2; e_hats = cell(epochs,1); x0s = cell(epochs,1);ks = 0:1:h_stab-1;
%     figure(), grid on, hold on
%     for e_idx = 1:epochs
%         x0 = low + (high -low).*rand(nx,1); 
%         [e_hat, x] = d_sim(sys,zeros(h_stab,ne),x0,@tanh); 
%         x0s{e_idx,1} = x0;e_hats{e_idx,1} = e_hat;
%         plot(vecnorm(x,2,2)), plot(m_rho(ks,x0), '--')
%     end


    %% analyze l2 stability
%     L1 = [eye(nx),zeros(nx,nd), zeros(nx,nw);
%         A, B, B2];
%     L2 = [zeros(nd, nx), eye(ne), zeros(nd, nw);
%         C, D, D12];
% 
%     if b_gen
%         L3 = [zeros(nw,nx), zeros(nw,nd), eye(nw);
%             C2-H, D21, zeros(nz,nw)];
%     else
%         L3 = [zeros(nw,nx),zeros(nw,nd) eye(nw);
%             C2, D21, zeros(nz,nw)];
%     end
% 
%     M = @(ga) L1' * [-X, zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
%         L2' * [-ga^2*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)] * L2 + ...
%         L3' * P * L3;
% 
%     gas = logspace(0,6,100);
%     max_real_eig = zeros(length(gas), 1);
%     for idx = 1:length(gas)
%         max_real_eig(idx,1) = max(real(eig(M(gas(idx)))));
%     end
%     fprintf('max real eig M: %f\n',min(max_real_eig))
% 
% %     figure(), semilogx(gas, max_real_eig), grid on
%     
%     if b_gen
%         fprintf('max real eig (H^T H - X): %f\n',max(real(eig(H'*H-X))))
%     end
    %% find an upper bound on the l2 gain
    eps=1e-5;
    ga2s = [];
    L1 = [eye(nx),zeros(nx,nd), zeros(nx,nw);
        A, B, B2];
    L2 = [zeros(nd,nx), eye(nd), zeros(nd,nw);
        C, D, D12];

%     L = diag(lambda); % can be replaced by diag multiplier
    P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];
%     L = eye(nw);
    P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

    if (b_gen) || contains(model_name, 'dzn')
        X = sdpvar(nx,nx);
        lambda = sdpvar(nw,1);
        L = diag(lambda);
        ga2 = sdpvar(1,1);
        fprintf('---generalized sector bounds evaluation %s ---\n', model_name)
        H_tilde = sdpvar(nz,nx);
        L3 = [zeros(nw,nx), zeros(nw,nd), eye(nw);
            L * C2-H_tilde, L*D21, zeros(nz,nw)];
%         add_constr = ([-X, H';H, -eye(nx)]<= -eps*eye(2*nx));
        add_constr = [];
        lmis = [];
        lmi = L1' * [-X, zeros(nx,nx); zeros(nx,nx), X] * L1 + ...
            L2' * [-ga2*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)] * L2 + ...
            L3' * [-(L+L'), eye(nw); eye(nw), zeros(nw,nw)] * L3;
    %         L3' * P * L3;
    
        lmis = lmis + (lmi <= -eps * eye(size(L1,2)));
        lmis = lmis + (X >= eps*eye(nx));
        lmis = lmis + add_constr;
         
        sol = optimize(lmis, ga2, sdpsettings('solver','MOSEK','verbose', 0));
        if sol.problem == 0
            fprintf('parameters have optimal gamma: %g \n', sqrt(double(ga2)))
        else
            fprintf('Trained parameters are not feasible: %s \n', sol.info)
        end
        ga2s(end+1) = double(ga2);

    end
    
    X = sdpvar(nx,nx);
    lambda = sdpvar(nw,1);
    L=diag(lambda);
    ga2 = sdpvar(1,1);

    L3 = [zeros(nw,nx),zeros(nw,nd) eye(nw);
    L * C2, L * D21, zeros(nz,nw)];
    add_constr = [];

    lmis = [];
    lmi = L1' * [-X, zeros(nx,nx); zeros(nx,nx), X] * L1 + ...
        L2' * [-ga2*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)] * L2 + ...
        L3' * [-(L+L'), eye(nw); eye(nw), zeros(nw,nw)] * L3;
%         L3' * P * L3;

    lmis = lmis + (lmi <= -eps * eye(size(L1,2)));
    lmis = lmis + (X >= eps*eye(nx));
    lmis = lmis + add_constr;
     
    sol = optimize(lmis, ga2, sdpsettings('solver','MOSEK','verbose', 0));
    if sol.problem == 0
        fprintf('parameters have optimal gamma: %g \n', sqrt(double(ga2)))
    else
        fprintf('Trained parameters are not feasible: %s \n', sol.info)
    end
    ga2s(end+1) = double(ga2);

    clear("H_tilde")
    clear('H')

    % write l2 gain to validation log
    evals = jsondecode(fileread(evaluation_file_name));
    evals.additional_tests.stability_l2_upper_bound = struct('value',sqrt(min(ga2s)));
    fid = fopen(evaluation_file_name, 'w');
    fprintf(fid, '%s', jsonencode(evals,"PrettyPrint",true));
    fclose(fid);
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





