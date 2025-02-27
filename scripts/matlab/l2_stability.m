clear all, close all
% we want to test if stability implies finite gain l2-stability for
% sufficently large gamma

nx = 2; nw=1;nz=nw;nd=1;ne=1;
a = 0; b = 1;

multiplier_types = {'diag', 'none'};

%% first assume no multiplier
for mult_idx = 1:length(multiplier_types)
    multiplier = multiplier_types{mult_idx};
    fprintf('Multiplier type: %s\n', multiplier)
    switch multiplier
        case 'none'
            L = eye(nw);
            multiplier_constraints = [];
        case 'diag'
            la = sdpvar(nw,1);
            L = diag(la);
            multiplier_constraints = (L >= 0);
    end

    A_tilde = sdpvar(nx,nx);
    B2_tilde = sdpvar(nx,nw);

    C2_tilde = sdpvar(nz,nx);

    X = sdpvar(nx,nx);

    M11 = [-X, C2_tilde'; C2_tilde, -2*L];
    M21 = [A_tilde,B2_tilde];
    M = [M11,M21';M21,-X];

    constraints = [];

    constraints = constraints + (M<=-eps*eye(size(M,1)));
    constraints = constraints + multiplier_constraints;
    sol = optimize(constraints, [], sdpsettings('solver','MOSEK','verbose', 0));

    X = double(X);L = double(L);
    Xinv = X^(-1);
    A = Xinv * double(A_tilde);
    B2 = Xinv * double(B2_tilde);
    C2 = L^(-1) * double(C2_tilde);
    P_r = [-eye(nw) b*eye(nw); eye(nw) -a*eye(nw)];
    P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

    L1 = [eye(nx), zeros(nx,nw);
        A, B2];
    L3 = [zeros(nz,nx), eye(nw);
        C2, zeros(nz,nw)];
    M = L1' * [-X, zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
        L3' * P * L3;
    fprintf('max real eig M: %f\n', max(real(eig(M))))
    % use random matrices B, C, D, D12 and D21 and test if system is finite
    % gain stable
    epochs = 1;
    no_ell_2 = {};
    for idx = 1:epochs
%         B = rand(nx,nd);
%         C = rand(ne,nx); D = rand(ne,nd); D12=rand(ne,nw);
%         D21 = rand(nw,nd);
        B = zeros(nx,nd);
        C = zeros(ne,nx); D = zeros(ne,nd); D12=zeros(ne,nw);
        D21 = zeros(nw,nd);

        L1 = [eye(nx), zeros(nx,nd) zeros(nx,nw);
            A, B, B2];
        L2 = [zeros(nd,nx), eye(nd) zeros(nd,nw);
            C, D, D12];
        L3 = [zeros(nw,nx), zeros(nw,nd) eye(nw);
            C2, D21, zeros(nz,nw)];

        M = @(ga) L1' * [-X, zeros(nx,nx);zeros(nx,nx), X] * L1 + ...
            L2' * [-ga^2*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)]*L2 + ...
            L3' * P * L3;

        gas = logspace(0,6,100); max_real_eig = zeros(length(gas),1);
        for ga_idx = 1:length(gas)
            max_real_eig(ga_idx,1) = max(real(eig(M(gas(ga_idx)))));
        end
        figure(), semilogx(gas,max_real_eig), grid on
        [m,i] = min(max_real_eig);
        if min(max_real_eig)<0
            fprintf('%i/%i: first eigenvalue: %f negative eigenvalue at gamma:  %f\n', idx,epochs, max_real_eig(1), gas(i))
        else
            no_ell_2{end+1} = struct( ...
                'B', B, ...
                'C', C, 'D', D, 'D12', D12, ...
                'C2', C2, 'D21',D21); %#ok<SAGROW>
%             warning('No negative eigenvalue.')
        end
    end
    no_ell_2 = [no_ell_2{:}];
    fprintf('length of parameters with no finite l2 gain %i\n', length(no_ell_2))
end


