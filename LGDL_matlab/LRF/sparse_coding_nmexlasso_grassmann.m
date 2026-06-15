function alpha = sparse_coding_nmexlasso_grassmann(X_all, D, lambda)
% X_all: d x p x m
% D    : d x p x N
% alpha: N x m

[~, p, ~] = size(X_all);
N = size(D, 3);

% ---- 1) Projection kernel Gram matrices ----
% K_D(j,k)  = || D_j' D_k ||_F^2
% K_XD(j,i) = || D_j' X_i ||_F^2   (note: size N x m)
K_D  = grassmann_proj_kernel(D);          % N x N
K_XD = grassmann_proj_kernel(D, X_all)';   % N x m

% ---- 2) Whiten (Algorithm 2 style): K_D = U S U' ----
[U, S] = svd(K_D, 'econ');
s = diag(S);

% numerical stability
eps_s = 1e-10;
s = max(s, eps_s);

% A = S^{1/2} U^T / sqrt(p)
A = (diag(sqrt(s)) * U') / sqrt(p);        % N x N

% x* = S^{-1/2} U^T K(X,D) / sqrt(p)
Xstar = (diag(1./sqrt(s)) * U') * K_XD / sqrt(p);  % N x m

% ---- 3) LASSO:  min_a ||x - A a||^2 + (lambda/2)||a||_1 ----
% nmexLasso expects: X (N x m), D (N x N) -> returns (N x m)
param.mode   = 2;
param.lambda = double(lambda) / 2;

if exist('nmexLasso', 'file') ~= 2
    error('nmexLasso not found on MATLAB path.');
end

alpha = full(nmexLasso(Xstar, A, param));  % N x m
end

% ============================================================
function K = grassmann_proj_kernel(SY1, SY2)
% Projection kernel: K(b,a) = || Y2_b' * Y1_a ||_F^2
% If only SY1 is given, returns pairwise kernel within SY1.

MIN_THRESH = 1e-6;

if nargin < 2
    SY2 = SY1;
    same_flag = true;
else
    same_flag = false;
end

p = size(SY1, 2);
[~, ~, n1] = size(SY1);
[~, ~, n2] = size(SY2);

K = zeros(n2, n1);

if same_flag
    for a = 1:n1
        Y1 = SY1(:,:,a);
        for b = a:n2
            M = Y1' * SY2(:,:,b);
            v = sum(M(:).^2);
            if v < MIN_THRESH, v = 0; elseif v > p, v = p; end
            K(b,a) = v;
            K(a,b) = v;
        end
    end
else
    for a = 1:n1
        Y1 = SY1(:,:,a);
        for b = 1:n2
            M = Y1' * SY2(:,:,b);
            v = sum(M(:).^2);
            if v < MIN_THRESH, v = 0; elseif v > p, v = p; end
            K(b,a) = v;
        end
    end
end
end