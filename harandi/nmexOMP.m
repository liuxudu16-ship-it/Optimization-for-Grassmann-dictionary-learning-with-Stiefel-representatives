function [A, path] = nmexOMP(X, D, param)
% 纯 MATLAB 版 OMP，用来替代原作者的 mexOMP.mexw64
% Usage:
%   A = mexOMP(X, D, param);
%   [A, path] = mexOMP(X, D, param);
%
% X: m x n   输入信号（每列一个样本）
% D: m x p   字典（每列一个原子，建议单位范数）
% param.L   : 稀疏度上限
% param.eps : (可选) 残差平方范数阈值，满足时提前停止
%
% A: p x n   系数矩阵（每列对应 X 的一列）
% path: 这里不使用，返回 []，只是为了接口兼容

    if nargin < 3
        param = struct;
    end

    [m, n]   = size(X);
    [mD, p]  = size(D);
    if m ~= mD
        error('mexOMP: dimension mismatch between X and D.');
    end

    % 稀疏度 L
    if isfield(param, 'L') && ~isempty(param.L)
        L = param.L;
    else
        L = min(m, p);
    end

    % 残差阈值（可选）
    if isfield(param, 'eps') && ~isempty(param.eps)
        eps_tol = param.eps;
    else
        eps_tol = 0;
    end

    % 预分配输出
    A = zeros(p, n);

    % 对每个样本单独做 OMP
    for i = 1:n
        x = X(:, i);      % 当前信号
        r = x;            % 当前残差
        idx_set = [];     % 已选择的原子索引
        a = [];           % 当前非零系数

        for k = 1:L
            % 1) 选择与残差相关性最大的原子
            proj = D' * r;                    % p x 1
            [~, j] = max(abs(proj));          % 最大相关性的原子索引

            % 如果重复选中同一个原子，则停止
            if any(idx_set == j)
                break;
            end
            idx_set = [idx_set, j];

            % 2) 在当前选择的原子子字典上做最小二乘
            Dj = D(:, idx_set);               % m x |idx_set|
            a  = Dj \ x;                       % |idx_set| x 1

            % 3) 更新残差
            r = x - Dj * a;

            % 4) 检查残差阈值
            if eps_tol > 0 && (r' * r) <= eps_tol
                break;
            end
        end

        % 把系数写回到对应位置
        if ~isempty(idx_set)
            A(idx_set, i) = a;
        end
    end

    if nargout > 1
        path = [];  % 这里不使用，留空即可
    end
end
