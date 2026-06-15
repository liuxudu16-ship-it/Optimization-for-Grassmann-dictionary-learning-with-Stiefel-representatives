function [A, path] = nmexLasso(varargin)
% 纯 MATLAB 版 mexLasso，用来替代原作者的 mexLasso.mexw64
%
% 使用方式（与原版接口兼容）：
%   A = mexLasso(X, D, param);
%   A = mexLasso(X, D, Q, param);   % Q 将被忽略，仅为接口兼容
%   [A, path] = mexLasso(...);     % path 返回 []，仅占位
%
% X : m x n 输入信号（每列一个样本）
% D : m x p 字典（每列一个原子）
% param.lambda : L1 正则化系数（标量），必需
%
% 返回：
% A   : p x n 稀疏系数矩阵（sparse double）
% path: 保留为空 []，仅为了接口兼容

    % -------- 参数解析 --------
    if nargin == 3
        X     = varargin{1};
        D     = varargin{2};
        param = varargin{3};
    elseif nargin == 4
        X     = varargin{1};
        D     = varargin{2};
        % Q    = varargin{3};  % Gram 矩阵，这里不使用
        param = varargin{4};
    else
        error('mexLasso: invalid number of input arguments.');
    end

    X = real(double(X));
    D = real(double(D));

    [m, n]  = size(X);
    [mD, p] = size(D);
    if m ~= mD
        error('mexLasso: dimension mismatch between X and D.');
    end

    if ~isfield(param, 'lambda') || isempty(param.lambda)
        error('mexLasso: param.lambda (L1 regularization weight) must be provided.');
    end

    lambda = param.lambda;
    if numel(lambda) > 1
        % 为简单起见，如果是向量，就取第一个值
        lambda = lambda(1);
    end
    lambda = double(lambda);

    % -------- 为所有列做 LASSO --------
    A = sparse(p, n);   % 输出系数

    for i = 1:n
        yi = X(:, i);        % m x 1
        yi = real(double(yi));

        % MATLAB 自带 lasso: 目标是 0.5||yi - D*alpha||^2 + lambda*||alpha||_1
        % 这里取对应 lambda 的那一列
        % 注意：lasso 的输入格式是 (观测数 × 特征数)
        try
            B = lasso(D, yi, ...
                      'Lambda', lambda, ...
                      'Standardize', false, ...
                      'RelTol', 1e-4, ...
                      'MaxIter', 1e3);
            % B: p x 1 （因为 Lambda 只有一个）
            A(:, i) = sparse(B);
        catch ME
            warning('mexLasso: lasso failed at column %d (%s). Setting coefficients to zero.', ...
                    i, ME.message);
            % 对这一列保持全零
        end
    end

    if nargout > 1
        path = [];  % 占位，原 MEX 里有 path 输出，这里不用
    end
end
