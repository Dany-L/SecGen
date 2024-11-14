function [e_hat_n, x] = d_sim(sys, d_n,x0, nl)
    h = size(d_n,1); nx = size(sys.A,1); nw=size(sys.B2,2); nz = size(sys.C2,1); ne = size(sys.C,1);
    x = zeros(h+1,nx);w = zeros(h,nw);e_hat_n = zeros(h,ne);z =zeros(h,nz);
    x(1,:) = x0; % initial condition
    for k=1:h
        z(k,:) = sys.C2 * x(k,:)' + sys.D21 * d_n(k,:)';
        w(k,:) = nl(z(k,:));
        x(k+1,:) = sys.A * x(k,:)' + sys.B * d_n(k,:)' + sys.B2 * w(k,:)';
        e_hat_n(k,:)= sys.C * x(k,:)' + sys.D * d_n(k,:)' + sys.D12 * w(k,:)';
    end
end