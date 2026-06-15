function T = compute_T(S)
% compute_T  Construct the T matrix used in SVD/Grassmann adjoint.
%
% Definition (Townsend 2016, eq.11):
%   T_ij = 1/(s_j^2 - s_i^2) for i ~= j, and 0 on diagonal.
%
% Input:
%   S : vector (p×1) of singular values, OR diagonal matrix (p×p).
%
% Output:
%   T : (p×p) matrix with zero diagonal.

    % ---- get singular values as a column vector s ----
    if ismatrix(S) && ~isvector(S)
        s = diag(S);
    else
        s = S(:);
    end

    p = length(s);

    % ---- build pairwise denominator s_j^2 - s_i^2 ----
    s2 = s.^2;
    denom = (s2.' - s2);   % denom(i,j) = s_j^2 - s_i^2

    % ---- avoid division on diagonal ----
    T = zeros(p,p);
    mask = ~eye(p);        % off-diagonal positions
    T(mask) = 1 ./ denom(mask);

    % ---- numeric safety: if denom is extremely small, set to 0 ----
    eps0 = 1e-12;
    T(abs(denom) < eps0) = 0;
end
