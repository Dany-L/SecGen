function z = sat(w)
%DZN Summary of this function goes here
%   Detailed explanation goes here
q = 1;
nw = size(w,2);
z = zeros(1,nw);

for w_idx =1:nw
    w_i = w(w_idx);
    if w_i > q
        z_i = q;
    elseif w_i < -q
        z_i = -q;
    else
        z_i = w_i;
    end
    z(w_idx) = z_i;

end

