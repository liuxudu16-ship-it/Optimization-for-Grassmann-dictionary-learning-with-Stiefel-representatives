function [D, alpha, cost_history] = train_grassmann_dict_gd(X_all, D0, lambda, T_out, step, nInner)
% Train Grassmann dictionary via alternating:
%   1) LASSO sparse coding (nmexLasso)
%   2) GD dictionary update on Stiefel/Grassmann
%
% X_all : d x p x m
% D0    : d x p x N
% alpha : N x m
% cost_history: 1 x T_out

D = D0;

% basic sizes
[~, p, m] = size(X_all);
N = size(D0, 3);

cost_history = zeros(1, T_out);

for t = 1:T_out
    % 1) sparse coding
    alpha = sparse_coding_nmexlasso_grassmann(X_all, D, lambda); % N x m

    % 2) dictionary update
    D = dict_update_gd_grassmann(X_all, D, alpha, step, nInner);

    % 3) cost (intrinsic to this file, no external dependency)
    cost_history(t) = compute_cost_projmat_local(X_all, D, alpha, lambda, p, m);
    fprintf('Iter %d/%d: cost=%.6g\n', t, T_out, cost_history(t));
end
end

% ============================================================
% Local cost (projection-matrix model) + L1 penalty
%   f = sum_i || X_i X_i^T - sum_j a_{j,i} D_j D_j^T ||_F^2 + lambda ||alpha||_1
% This equals:
%   p*m - 2*tr(alpha' K_DX) + tr(alpha' K_DD alpha) + lambda*|alpha|_1
% where K_DD(j,k)=||D_j^T D_k||_F^2, K_DX(j,i)=||D_j^T X_i||_F^2
% ============================================================
function cost = compute_cost_projmat_local(X, D, alpha, lambda, p, m)
K_DD = grassmann_proj_kernel_local(D);       % N x N
K_DX = grassmann_proj_kernel_local(D, X)';   % N x m  (NOTE transpose!)

cost = p*m - 2*trace(alpha' * K_DX) + trace(alpha' * K_DD * alpha);
cost = cost + lambda * sum(abs(alpha(:)));
end

% ============================================================
% Local kernel: K(i,j) = || A_i^T B_j ||_F^2 (clipped to [0,p])
% If only one input: returns N x N
% If two inputs: returns (#B) x (#A)  (so caller decides transpose)
% ============================================================
function K = grassmann_proj_kernel_local(SY1, SY2)
MIN_THRESH = 1e-6;

if nargin < 2
    SY2 = SY1;
    same = true;
else
    same = false;
end

p = size(SY1, 2);
n1 = size(SY1, 3);
n2 = size(SY2, 3);

K = zeros(n2, n1);

if same
    for i = 1:n1
        Y1 = SY1(:,:,i);
        for j = i:n2
            v = sum(sum((Y1' * SY2(:,:,j)).^2));
            if v < MIN_THRESH, v = 0; end
            if v > p, v = p; end
            K(j,i) = v;
            K(i,j) = v;
        end
    end
else
    for i = 1:n1
        Y1 = SY1(:,:,i);
        for j = 1:n2
            v = sum(sum((Y1' * SY2(:,:,j)).^2));
            if v < MIN_THRESH, v = 0; end
            if v > p, v = p; end
            K(j,i) = v;
        end
    end
end
end