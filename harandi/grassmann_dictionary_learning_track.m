function [D, cost_history] = grassmann_dictionary_learning_track(X, ~, dict_options, D0_h)
    % 初始化字典
    D = D0_h;

    % 外迭代次数
    nIter = dict_options.nIter;

    %--------------------------------------------------------------
    % 兼容旧版：cost_history 仍然保存"初始 + 每轮字典更新后"的 cost
    %--------------------------------------------------------------
    cost_history = nan(1, nIter + 1);

    %--------------------------------------------------------------
    % 新增：详细记录
    %--------------------------------------------------------------
    cost_detail = struct();
    cost_detail.init = [];
    cost_detail.after_sparse = nan(1, nIter);
    cost_detail.after_dict   = nan(1, nIter);

    %--------------------------------------------------------------
    % 初始 cost：与旧代码一致
    %--------------------------------------------------------------
    alpha_init = local_sparse_coding(X, D, dict_options);
    initCost = compute_dic_cost(X, D, alpha_init);

    cost_history(1) = initCost;
    cost_detail.init = initCost;

    fprintf('Initial cost --> %.6f\n', initCost);

    %--------------------------------------------------------------
    % 主循环
    %--------------------------------------------------------------
    for tmpIter = 1:nIter
        % 1) 稀疏编码
        alpha = local_sparse_coding(X, D, dict_options);

        sparseCost = compute_dic_cost(X, D, alpha);
        cost_detail.after_sparse(tmpIter) = sparseCost;
        fprintf('Iter#%d [After Sparse Coding] cost --> %.6f\n', tmpIter, sparseCost);

        % 2) 字典更新
        Dn = update_dict(X, D, alpha);

        dictCost = compute_dic_cost(X, Dn, alpha);
        cost_detail.after_dict(tmpIter) = dictCost;

        % 兼容旧版：cost_history 只记录"字典更新后"的 cost
        cost_history(tmpIter + 1) = dictCost;

        fprintf('Iter#%d [After Dict Update ] cost --> %.6f\n', tmpIter, dictCost);

        % 更新字典
        D = Dn;
    end

    fprintf('-------\n');
end

%--------------------------------------------------------------------------
function Dn = update_dict(X, D, alpha)
% 严格按图中伪代码实现的顺序字典更新
%
% for r = 1,...,nAtoms
%   S_r = sum_i alpha(r,i) * ( Xhat_i
%                              - sum_{j<r} alpha(j,i) * Dhat_j^k
%                              - sum_{j>r} alpha(j,i) * Dhat_j^{k-1} )
%   取 S_r 的前 p 个特征向量
%   设 D_r^k = [v1, ..., vp]
% end
%
% 这里：
%   Xhat_i      = X(:,:,i) * X(:,:,i)'
%   Dhat_j^k    = Dn(:,:,j) * Dn(:,:,j)'   (j < r)
%   Dhat_j^{k-1}= D(:,:,j)  * D(:,:,j)'    (j > r)

    sym_mat = @(A) real(0.5 * (A + A'));

    nAtoms = size(D, 3);
    [n, p, ~] = size(X);

    % Dn 保存本轮第 k 轮更新后的字典
    % 初始先放旧字典，后面按 r=1,...,nAtoms 逐个覆盖
    Dn = D;

    for r = 1:nAtoms
        S = zeros(n);

        % 只遍历 alpha(r,i) 非零的样本；与全求和等价，只是更高效
        idx_alpha = find(alpha(r, :) ~= 0);

        for t = 1:length(idx_alpha)
            i = idx_alpha(t);

            % \hat{X}_i
            Xi_hat = X(:,:,i) * X(:,:,i)';

            % 累加 alpha(r,i) * \hat{X}_i
            S = S + alpha(r, i) * Xi_hat;

            % 减去 sum_{j<r} alpha(r,i)*alpha(j,i) * \hat{D}_j^k
            for j = 1:r-1
                if alpha(j, i) == 0
                    continue;
                end
                S = S - alpha(r, i) * alpha(j, i) * (Dn(:,:,j) * Dn(:,:,j)');
            end

            % 减去 sum_{j>r} alpha(r,i)*alpha(j,i) * \hat{D}_j^{k-1}
            for j = r+1:nAtoms
                if alpha(j, i) == 0
                    continue;
                end
                S = S - alpha(r, i) * alpha(j, i) * (D(:,:,j) * D(:,:,j)');
            end
        end

        S = sym_mat(S);

        % 取前 p 个最大特征值对应特征向量
         [Ur, ~] = eigs(S, p, 'LA');

        % 立即写回本轮第 k 轮的新原子 D_r^k
        Dn(:,:,r) = Ur;
    end
end
%--------------------------------------------------------------------------
function cost = compute_dic_cost(X, D, alpha)
[~, p, nPoints] = size(X);
k_DD = p - 0.5*grassmann_proj_dist(D);
k_DX = p - 0.5*grassmann_proj_dist(X, D);
cost = p*nPoints - 2*trace(alpha'*k_DX) + trace(alpha'*k_DD*alpha);
end

%--------------------------------------------------------------------------
function centers = kmeans_projection(X, k, nIter, verbatim_flag)
MinCostVariation = 1e-3;
nPoints = size(X, 3);

randVal = randperm(nPoints);
centers = X(:,:,randVal(1:k));
preCost = 0;

for iter = 1:nIter
    fprintf('.');
    [currCost, minIdx] = kmeans_cost(X, centers);
    for tmpC1 = 1:k
        idx = find(minIdx == tmpC1);
        if (isempty(idx))
            randVal = randperm(nPoints);
            centers(:,:,tmpC1) = X(:,:,randVal(1));
        else
            centers(:,:,tmpC1) = grassmann_mean_proj_local(X(:,:,idx));
        end
    end
    if (iter > 1)
        cost_diff = norm(preCost - currCost);
        if (cost_diff < MinCostVariation)
            break;
        end
    end
    preCost = currCost;
end
fprintf('\n');
end

%--------------------------------------------------------------------------
function [outCost, minIdx] = kmeans_cost(X, centers)
l_dist = grassmann_proj(X, centers);
[minDist, minIdx] = max(l_dist);
outCost = sum(minDist);
end

%--------------------------------------------------------------------------
function outMean = grassmann_mean_proj_local(X)
[n, p, nPoints] = size(X);
tmpBig = zeros(n, n);
for tmpC1 = 1:nPoints
    tmpBig = tmpBig + X(:,:,tmpC1)*X(:,:,tmpC1)';
end
[outMean, ~] = eigs(tmpBig, p);
end

%--------------------------------------------------------------------------
function dist_p = grassmann_proj(SY1, SY2)
MIN_THRESH = 1e-6;
same_flag = false;
if (nargin < 2)
    SY2 = SY1;
    same_flag = true;
end
p = size(SY1, 2);

[~, ~, number_sets1] = size(SY1);
[~, ~, number_sets2] = size(SY2);

dist_p = zeros(number_sets2, number_sets1);

if (same_flag)
    for tmpC1 = 1:number_sets1
        Y1 = SY1(:,:,tmpC1);
        for tmpC2 = tmpC1:number_sets2
            tmpMatrix = Y1' * SY2(:,:,tmpC2);
            tmpProjection_Kernel_Val = sum(sum(tmpMatrix.^2));
            if (tmpProjection_Kernel_Val < MIN_THRESH)
                tmpProjection_Kernel_Val = 0;
            elseif (tmpProjection_Kernel_Val > p)
                tmpProjection_Kernel_Val = p;
            end
            dist_p(tmpC2, tmpC1) = tmpProjection_Kernel_Val;
            dist_p(tmpC1, tmpC2) = dist_p(tmpC2, tmpC1);
        end
    end
else
    for tmpC1 = 1:number_sets1
        Y1 = SY1(:,:,tmpC1);
        for tmpC2 = 1:number_sets2
            tmpMatrix = Y1' * SY2(:,:,tmpC2);
            tmpProjection_Kernel_Val = sum(sum(tmpMatrix.^2));
            if (tmpProjection_Kernel_Val < MIN_THRESH)
                tmpProjection_Kernel_Val = 0;
            elseif (tmpProjection_Kernel_Val > p)
                tmpProjection_Kernel_Val = p;
            end
            dist_p(tmpC2, tmpC1) = tmpProjection_Kernel_Val;
        end
    end
end
end

%--------------------------------------------------------------------------
function alpha = local_sparse_coding(X, dicX, dict_options)
[~, p, ~] = size(dicX);
if isstruct(dict_options)
    if ~isfield(dict_options,'L')
        dict_options.L = 10;
    end
    if ~isfield(dict_options,'coding')
        dict_options.coding = 'omp';
    end
    if ~isfield(dict_options,'lambda')
        dict_options.lambda = 1e-3;
    end
    L = dict_options.L;
    coding = dict_options.coding;
    lambda = dict_options.lambda;
else
    L = dict_options;
    coding = 'omp';
    lambda = 1e-3;
end

K_D = grassmann_proj(dicX);
K_XD = grassmann_proj(X, dicX);

[KD_U, KD_D, ~] = svd(K_D);
D = diag(sqrt(diag(KD_D)))*KD_U'/sqrt(p);
D_Inv = KD_U*diag(1./sqrt(diag(KD_D)));
qX = D_Inv'*K_XD/sqrt(p);
try
    if strcmpi(coding, 'lasso')
        if numel(lambda) > 1
            lambda = lambda(1);
        end
        lambda = max(double(lambda), 1e-12);
        param = struct('lambda', lambda);
        if exist('mexLasso', 'file') == 2 || exist('mexLasso', 'file') == 3
            alpha = full(mexLasso(qX, D, param));
        else
            alpha = full(nmexLasso(qX, D, param));
        end
    else
        if exist('mexOMP', 'file') == 2 || exist('mexOMP', 'file') == 3
            alpha = full(mexOMP(qX, D, struct('L', L)));
        else
            alpha = full(nmexOMP(qX, D, struct('L', L)));
        end
    end
catch ME
    warning('Harandi:SparseCoding_failed', 'Sparse coding failed: %s. Using zeros.', ME.message);
    alpha = zeros(size(D, 2), size(qX, 2));
end
end

