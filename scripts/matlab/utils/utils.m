
classdef utils

    methods(Static)

        function L = lower_triangular_from_vector(L_flat, nx)
            % construct lower triangular matrix from vector.
            L = zeros(nx,nx);flat_idx = 0;
            for diag_idx = -0:-1:-nx+1
                diag_len = diag_idx + nx;
                L = L + diag(L_flat(flat_idx+1: flat_idx + diag_len),diag_idx);
                flat_idx = flat_idx + diag_len;
            end
        end

        %         function
        %
        %
        %             def construct_lower_triangular_matrix(
        %         self, L_flat: torch.Tensor, diag_length: int
        %     ) -> torch.Tensor:
        %         device = L_flat.device
        %         flat_idx = 0
        %         L = torch.zeros(
        %             size=(diag_length, diag_length), dtype=torch.float32, device=device
        %         )
        %         for diag_idx, diag_size in zip(
        %             range(0, -diag_length, -1), range(diag_length, 0, -1)
        %         ):
        %             L += torch.diag(L_flat[flat_idx : flat_idx + diag_size], diagonal=diag_idx)
        %             flat_idx += diag_size
        %
        %         return L
        %
        %     def extract_vector_from_lower_triangular_matrix(
        %         self, L: NDArray[np.float64]
        %     ) -> NDArray[np.float64]:
        %         diag_length = L.shape[0]
        %         vector_list = []
        %         for diag_idx in range(0, -diag_length, -1):
        %             vector_list.append(np.diag(L, k=diag_idx))
        %
        %         return np.hstack(vector_list)
        function [es, ds]= load_data_from_dir(directory, input_names, output_names, filter)
            if nargin < 3
                filter = false;
            end
            filenames = dir(directory);
            es = {}; ds = {};
            for i=1:length(filenames)
                filename = filenames(i).name;
                [~,~,ext] = fileparts(filename);
                if ~strcmp(ext,'.csv')
                    continue
                end

                if filter
                    if ~contains(filename, filter)
                        continue
                    end
                end
                tab = readtable(fullfile(directory,filename));

                ds{end+1} = utils.get_sequence_from_tab(tab, input_names); %#ok<AGROW>
                es{end+1} = utils.get_sequence_from_tab(tab, output_names); %#ok<AGROW>


            end
        end

        function seq = get_sequence_from_tab(tab, names)
            N = size(tab,1); n = length(names);
            seq = zeros(N, n);
            for s_idx=1:length(names)
                seq(:,s_idx) = tab.(names{s_idx});
            end
        end

        function norm_data_cell = normalize_cell(data_cell, mean, std)
            N = length(data_cell); norm_data_cell = cell(N,1);
            for idx = 1:N
                norm_data_cell{idx} = utils.normalize_(data_cell{idx}, mean, std);
            end
        end

        function n_data = normalize_(data, mean, std)
            n_data = (data - mean)./std;
        end

        function data = denormalize_(n_data, mean, std)
            data = n_data .* std + mean;
        end

        function [m, s] = get_mean_std(data_cell)
            stacked_data = cat(1,data_cell{:});
            m = mean(stacked_data); s = std(stacked_data);
        end

                function data = read_data_from_tab(tab,column_names)
            data = [];
            for idx =1:length(column_names)
                column_name = column_names{idx};
                data = [data, tab.(column_name)];
            end

        end

        function [d_f,e_f,ts] = read_data_from_filename(filename, input_names, output_names)
            tab = readtable(filename);
            try
                ts = tab.time(2)-tab.time(1);
            catch
                ts = 0;
            end
            e_f = utils.read_data_from_tab(tab,output_names);
            d_f = utils.read_data_from_tab(tab,input_names);
        end

        function [d,e, ts] = read_data_from_filenames(filenames, directory, input_names, output_names)
            d = [];
            e = [];
            for idx = 1:length(filenames)
                filename = filenames(idx).name;
                if any(strcmp(filename, {'.', '..'}))
                    continue
                end
                [d_f,e_f,ts] = read_data_from_filename(fullfile(directory, filename), input_names, output_names);
                d = [d; d_f];
                e = [e; e_f];
            end
        end

        function data_norm = normalize(data,mean,std)
            data_norm = zeros(size(data));
            for idx = 1:size(data ,2)
                data_norm(:,idx) = (data(:,idx)-mean(idx))/std(idx);
            end
        end

        function data = denormalize(data_norm,mean,std)
            data = zeros(size(data_norm));
            for idx = 1:size(data_norm ,2)
                data(:,idx) = data_norm(:,idx) * std(idx) + mean(idx);
            end
        end

        function gamma = hinfnorm_own(sys)
            nx = size(sys.A,1);
            nd = size(sys.B,2);
            ne = size(sys.C,1);

            ga = sdpvar(1,1);
            X = sdpvar(nx,nx);

            L1 = [eye(nx), zeros(nx,nd);
                sys.A,sys.B];
            L2 = [zeros(nd,nx), eye(nd);
                sys.C,sys.D];
            lmis = [];
            lmis = lmis + ...
                (L1'*[-X, zeros(nx,nx);zeros(nx,nx), X]*L1 ...
                + L2'*[-ga*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)]*L2<=0);
            sol = optimize(lmis,ga,sdpsettings('solver','MOSEK','verbose', 0));
            gamma = sqrt(double(ga));
        end

        function ss = load_ss_from_json(filename)
            sys_struct = jsondecode(fileread(filename));
            nx = size(sys_struct.A_lin,1);
            ss = dss(sys_struct.A_lin,sys_struct.B_lin,sys_struct.C_lin,sys_struct.D_lin,eye(nx), sys_struct.ts);
        end

        function [e_hat, x] = sim_model(model,d,x0)
            ne = size(model.C,1);
            nxi = size(model.A,1);
            N = size(d,2);
            x = zeros(N+1,nxi);
            x(1,:) = x0;
            e_hat = zeros(N,ne);
            for k = 1:N
                d_k = d(:,k);
                x_k = x(k,:)';
        
                z_k = model.C2*x_k+ model.D21 * d_k;
                w_k = tanh(z_k);
                x(k+1,:) = model.A*x_k + model.B*d_k + model.B2 * w_k;
                e_hat(k,:) = model.C * x_k + model.D*d_k+ model.D12*w_k;
            end
        end


    end
end
