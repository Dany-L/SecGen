clear all;

%% setup
base_path = fileparts(mfilename('fullpath'));
% system_name = 'coupled-msd';
% training_directory = sprintf('~/%s/data/%s-routine/processed/train/',system_name, system_name);
% validation_direcotry = sprintf('~/%s/data/%s-routine/processed/validation/', system_name, system_name);
% input_names = {'u_1'};
% output_names = {'y_1'};
% ts = 0.2;

system_name = 'f16-gvt';
training_directory = sprintf('~/%s/data/F16GVT_Files/BenchmarkData',system_name, system_name);
validation_directory = sprintf('~/%s/data/F16GVT_Files/BenchmarkData',system_name, system_name);
input_names = {'Force'};
output_names = {'Acceleration1','Acceleration2','Acceleration3'};
ts = 1/400;


%% load data
[es_train, ds_train] = utils.load_data_from_dir(training_directory, input_names,output_names);
[es_val, ds_val] = utils.load_data_from_dir(validation_direcotry, input_names,output_names, 'Validation');
N = size(es_train{1},1);
%% normalize data
[d_mean, d_std] = utils.get_mean_std(ds_train);
[e_mean, e_std] = utils.get_mean_std(es_train);
ds_norm_train = utils.normalize_cell(ds_train,d_mean,d_std);
es_norm_train = utils.normalize_cell(es_train,e_mean,e_std);
%% n4sid
nx = 8;
n_train_data = iddata(cat(1,es_norm_train{:}),cat(1,ds_norm_train{:}),ts);
% n4sidOptions('Focus', 'simulation', 'EnforceStability', 1)
sys = n4sid(n_train_data, nx);
A = sys.A;B=sys.B; C =sys.C; D=sys.D; K = sys.K;
sys_struct = struct('A', A, 'B',B, 'C', C, 'D', C, 'K', K);
%% system analysis and evaluation on validation set
[mag, phase, w] = bode(sys);

t = linspace(0,(N-1)*ts,N);
e = 0; K = length(es_val);
for val_idx = 1:length(ds_val)
    d_val_norm = utils.normalize_(ds_val{val_idx},d_mean,d_std); e_val = es_val{val_idx};
    e_hat_val = utils.denormalize_(lsim(sys,d_val_norm,t,zeros(nx,1)),e_mean,e_std);
    e = e + sqrt(1/(N*e_std.^2)*sum(e_hat_val-e_val).^2);
end
n_total_error = e/K;
fprintf('Total error: %f \n', n_total_error);

% save one output sequence
d_val_norm = utils.normalize_(ds_val{1},d_mean,d_std);
e_hat_val = utils.denormalize_(lsim(sys,d_val_norm,t,zeros(nx,1)),e_mean,e_std);
filename = sprintf('n4sid_mat-%s-nx_%d.mat', system_name, nx);
fields = fieldnames(sys_struct);
save(fullfile(base_path,'data',filename), 'mag', 'phase', 'w','e_hat_val','n_total_error', fields{:})
plot(t,e_hat_val); hold on, grid on
plot(t,es_val{1}, '--'), legend({'$\hat{e}$', '$e$'},'interpreter','latex')











