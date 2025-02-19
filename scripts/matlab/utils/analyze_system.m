function l2_gain = analyze_system(sys, alpha, beta, H)
    if H == false
        b_gen = false;
    else
        b_gen= true;
    end
        

    nx = size(sys.A,2); nd = size(sys.B,2); nw = size(sys.B2,2); 
    ne = size(sys.C,1); nz = nw;

    l2_gain = -1;
    eps=1e-5;

    X = sdpvar(nx,nx);
    lambda = sdpvar(nw,1);
    ga2 = sdpvar(1,1);

    L1 = [eye(nx),zeros(nx,nd), zeros(nx,nw);
        sys.A, sys.B, sys.B2];
    
    L2 = [zeros(nd,nx), eye(nd), zeros(nd,nw);
        sys.C, sys.D, sys.D12];

    if b_gen
        % H = sdpvar(nz,nx);
        L3 = [zeros(nw,nx), zeros(nw,nd), eye(nw);
            sys.C2-H, sys.D21, zeros(nz,nw)];    
        add_constr = ([-X, H';H, -eye(nz)]<= -eps*eye(2*nx));
        % add_constr = [];
    else
        L3 = [zeros(nw,nx), zeros(nw,nd), eye(nw);
            sys.C2, sys.D21, zeros(nz,nw)];
        add_constr = [];
    end

    P_r = [-eye(nw) beta*eye(nw); eye(nw) -alpha*eye(nw)];
    L = diag(lambda); % can be replaced by diag multiplier
    P = P_r' * [zeros(nw,nw), L'; L, zeros(nw,nw)] * P_r;

    lmis = [];
    lmi = L1' * [-X, zeros(nx,nx); zeros(nx,nx), X] * L1 + ...
        L2' * [-ga2*eye(nd), zeros(nd,ne); zeros(ne,nd), eye(ne)] * L2 + ...
        L3' * P * L3;

    lmis = lmis + (lmi <= -eps * eye(size(L1,2)));
    % lmis = lmis + (X >= eps*eye(nx));
    lmis = lmis + add_constr;
     
    sol = optimize(lmis, ga2, sdpsettings('solver','MOSEK','verbose', 0));
    if sol.problem == 0
        fprintf('parameters have optimal gamma: %g \n', sqrt(double(ga2)))
    else
        fprintf('Trained parameters are not feasible: %s \n', sol.info)
        return
    end

    l2_gain = sqrt(double(ga2));