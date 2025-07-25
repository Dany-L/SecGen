clear all;

%% setup
base_path = fileparts(mfilename('fullpath'));
% system_name = 'coupled-msd';
% training_directory = sprintf('~/%s/data/%s-routine/processed/train/',system_name, system_name);
% validation_direcotry = sprintf('~/%s/data/%s-routine/processed/validation/', system_name, system_name);
% input_names = {'u_1'};
% output_names = {'y_1'};
% ts = 0.2;

system_name = 'F16';
training_directory = sprintf('~/%s/data/F16GVT_Files/BenchmarkData',system_name);
validation_directory = sprintf('~/%s/data/F16GVT_Files/BenchmarkData',system_name);
    input_names = {'Force'};
    output_names = {'Acceleration1','Acceleration2','Acceleration3'};
ts = 1/400;


%% load data
[es_train, ds_train] = utils.load_data_from_dir(training_directory, input_names,output_names, {''}, {'Validation', 'SpecialOdd'});
[es_val, ds_val] = utils.load_data_from_dir(validation_directory, input_names,output_names, {'Validation'}, {'SpecialOdd'});
% print number of training samples
n_train = 0; for idx = 1:length(es_train), n_train=n_train+size(es_train{idx},1); end, fprintf('Number of training samples %d\n', n_train)
N = size(es_train{1},1);
%% normalize data
[d_mean, d_std] = utils.get_mean_std(ds_train);
[e_mean, e_std] = utils.get_mean_std(es_train);
ds_norm_train = utils.normalize_cell(ds_train,d_mean,d_std);
es_norm_train = utils.normalize_cell(es_train,e_mean,e_std);
%% n4sid
nx = 8;
n_train_data = iddata(cat(1,es_norm_train{:}),cat(1,ds_norm_train{:}),ts);
n4sidOptions('Focus', 'simulation', 'InitialState','zero');
tic;
disturbance_model = 'none';
sys = n4sid(n_train_data, nx, 'DisturbanceModel',disturbance_model);
elapsed_time = toc;  % Time in seconds
A = sys.A;B=sys.B; C =sys.C; D=sys.D; K = sys.K;

% Build struct to store
sys_struct = struct();
sys_struct.A = A;
sys_struct.B = B;
sys_struct.C = C;
sys_struct.D = D;
sys_struct.K = K;
sys_struct.nx = nx;
sys_struct.ts = ts;
sys_struct.num_samples = n_train;
sys_struct.is_stable = isstable(sys);
sys_struct.n4sid_info = struct(sys.Report);
sys_struct.elapsed_time_sec = elapsed_time;

% Convert to HH:MM:SS using duration
elapsed_duration = duration(0, 0, elapsed_time);
elapsed_time_str = char(elapsed_duration);  % Convert to string for display

fprintf('n4sid identification completed in %s (HH:MM:SS)\n', elapsed_time_str);
% Save to MAT file
save(sprintf('./data/%s_%s_n4sid.mat',system_name, disturbance_model), 'sys_struct');

% obtain settling time is that equal to transient time?
step(sys)


%% system analysis and evaluation on validation set
% [mag, phase, w] = bode(sys);
% 
% t = linspace(0,(N-1)*ts,N);
% e = 0; K = length(es_val);
% for val_idx = 1:length(ds_val)
%     d_val_norm = utils.normalize_(ds_val{val_idx},d_mean,d_std); e_val = es_val{val_idx};
%     e_hat_val = utils.denormalize_(lsim(sys,d_val_norm,t,zeros(nx,1)),e_mean,e_std);
%     e = e + sqrt(1/(N*e_std.^2)*sum(e_hat_val-e_val).^2);
% end
% n_total_error = e/K;
% fprintf('Total error: %f \n', n_total_error);
% 
% % save one output sequence
% d_val_norm = utils.normalize_(ds_val{1},d_mean,d_std);
% e_hat_val = utils.denormalize_(lsim(sys,d_val_norm,t,zeros(nx,1)),e_mean,e_std);
% filename = sprintf('n4sid_mat-%s-nx_%d.mat', system_name, nx);
% fields = fieldnames(sys_struct);
% save(fullfile(base_path,'data',filename), 'mag', 'phase', 'w','e_hat_val','n_total_error', fields{:})
% plot(t,e_hat_val); hold on, grid on
% plot(t,es_val{1}, '--'), legend({'$\hat{e}$', '$e$'},'interpreter','latex')











