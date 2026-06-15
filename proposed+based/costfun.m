function [J, E_mat, G_mat, G] = costfun(LogX, LogD, Y)
[m, N] = size(Y);
[d, p] = size(LogD{1});
dp = d * p;

% ============================================================
% Persistent cache for LogX_mat (since LogX is constant during learning)
% ============================================================
persistent LogX_mat_cache cache_valid cache_m cache_dp cache_key

if isempty(cache_valid)
    cache_valid = false;
end

sig1 = sum(LogX{1}(:));
sig2 = sum(LogX{end}(:));
key  = [m, dp, sig1, sig2];

if ~cache_valid || cache_m ~= m || cache_dp ~= dp || any(cache_key ~= key)
    % build LogX_mat once
    LogX_mat_cache = zeros(dp, m);
    for i = 1:m
        LogX_mat_cache(:, i) = LogX{i}(:);
    end
    cache_valid = true;
    cache_m = m;
    cache_dp = dp;
    cache_key = key;
end

LogX_mat = LogX_mat_cache;

% ============================================================
% Build LogD_mat (dictionary changes frequently => no cache here)
% ============================================================
LogD_mat = zeros(dp, N);
for j = 1:N
    LogD_mat(:, j) = LogD{j}(:);
end

% ============================================================
% Residual & cost
% ============================================================
E_mat = LogX_mat - LogD_mat * Y';   % dp x m
J = sum(E_mat(:).^2);              % same as norm(E,'fro')^2 but faster

% ============================================================
% Gradient
% ============================================================
if nargout > 2
    % dp x N
    G_mat = -2 * E_mat * Y;

    % return G as cell(1,N)
    G = cell(1, N);
    for j = 1:N
        G{j} = reshape(G_mat(:, j), d, p);
    end
end

end
