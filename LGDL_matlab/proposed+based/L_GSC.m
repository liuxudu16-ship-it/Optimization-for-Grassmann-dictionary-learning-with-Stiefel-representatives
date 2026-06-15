function Y = L_GSC(LogX, LogD, lambda)
m = numel(LogX);
N = numel(LogD);
[d, p] = size(LogX{1});

% Dictionary (dp x N)
D_log = zeros(d*p, N);
for j = 1:N
    D_log(:, j) = double(LogD{j}(:));
end

% Data (dp x m)
X_log = zeros(d*p, m);
for i = 1:m
    X_log(:, i) = LogX{i}(:);
end

param.lambda = double(lambda) / 2;
param.mode = 2;

if exist('mexLasso', 'file') == 2 || exist('mexLasso', 'file') == 3
    A = mexLasso(X_log, D_log, param);   % N x m
else
    if exist('nmexLasso', 'file') ~= 2
        local_root = fileparts(mfilename('fullpath'));
        alt_dir = fullfile(local_root, '对比');
        if exist(alt_dir, 'dir'), addpath(alt_dir); end
    end
    if exist('nmexLasso', 'file') ~= 2
        error('mexLasso/nmexLasso not found on MATLAB path.');
    end
    A = nmexLasso(X_log, D_log, param);
end

Y = full(A).';  % m x N
end
