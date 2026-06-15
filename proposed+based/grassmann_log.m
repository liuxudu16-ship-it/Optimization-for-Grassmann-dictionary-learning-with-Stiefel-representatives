function [Delta, cache] = grassmann_log(U, Y)
% grassmann_log  Extended Grassmann logarithm (Algorithm 1) - STABLE VERSION
%
% Input:
%   U, Y : d×p orthonormal basis matrices (Stiefel reps)
%
% Output:
%   Delta : d×p horizontal tangent vector at U
%   cache : struct storing SVD factors for differential_adjoint
%
% Usage:
%   Delta = grassmann_log(U,Y);                % 单输出也可以
%   [Delta, cache] = grassmann_log(U,Y);      % 双输出带 cache
%
% 稳定性改进：
%   - 限制 S2 奇异值范围，防止 asin(s) 当 s≈1 时不稳定
%   - 检测和修正数值异常

    % ===== 数值稳定性参数 =====
    epsilon = 1e-10;  % 数值稳定阈值
    
    % -------- Step 1: SVD of Y' * U --------
    % Y' U = Q1 * S1 * R1'
    [Q1, S1, R1] = svd(Y' * U, 'econ');
    
    % -------- Step 2: Procrustes alignment --------
    % Y_* = Y * (Q1 * R1')
    Y_star = Y * (Q1 * R1');
    
    % -------- Step 3: compact SVD of (I - U U') Y_star --------
    % M = (I - U U') Y_star = Q2 * S2 * R2'
    M = Y_star - U * (U' * Y_star);
    [Q2, S2, R2] = svd(M, 'econ');
    
    % ===== 关键改进：限制 S2 的奇异值范围 =====
    % 确保 s2 严格在 [0, 1) 区间内，防止 asin 计算不稳定
    s2 = diag(S2);          % singular values vector
    
    % 裁剪到安全范围
    s2_safe = min(s2, 1 - epsilon);  % 防止 s2 = 1 → asin(1) = π/2 导致后续不稳定
    s2_safe = max(s2_safe, 0);       % 防止负值（理论上不会，但保险起见）
    
    % 检测是否进行了裁剪（调试时有用）
    if any(s2 > (1 - 10*epsilon))
        % 可选：打印警告（如果你想静默运行，可以注释掉）
        % warning('grassmann_log: S2 singular values clamped (max=%.10f)', max(s2));
    end
    
    % 更新 S2 为安全版本
    S2 = diag(s2_safe);
    
    % -------- Step 4: Sigma = asin(diag(S2)) --------
    Sigma_vec = asin(s2_safe);   % 现在使用安全的奇异值
    
    % ===== 额外保护：检查 Sigma 的合理性 =====
    % Sigma 应该在 [0, π/2) 范围内
    if any(Sigma_vec > pi/2 - epsilon)
        % 理论上不应该发生，但如果发生了就裁剪
        Sigma_vec = min(Sigma_vec, pi/2 - epsilon);
    end
    
    Sigma = diag(Sigma_vec);
    
    % -------- Step 5: Delta = Q2 * Sigma * R2' --------
    Delta = Q2 * (Sigma * R2');
    
    % ===== 最终检查：确保 Delta 没有 NaN 或 Inf =====
    if any(isnan(Delta(:))) || any(isinf(Delta(:)))
        warning('grassmann_log: NaN/Inf detected in Delta, likely numerical issue');
        % 返回零向量作为安全回退（保持在切空间中）
        Delta = zeros(size(Delta));
    end
    
    % -------- cache for differential_adjoint --------
    if nargout > 1
        cache = struct();
        cache.Q1 = Q1;  cache.S1 = S1;  cache.R1 = R1;
        cache.Q2 = Q2;  cache.S2 = S2;  cache.R2 = R2;
        cache.Sigma = Sigma;   % diag matrix
        cache.U = U;           % optional
        cache.Y = Y;           % optional
        cache.Ystar = Y_star;  % optional
        
        % ===== 新增：标记是否进行了裁剪 =====
        cache.clamped = any(s2 > (1 - 10*epsilon));
        cache.s2_original_max = max(s2);  % 记录原始最大奇异值（调试用）
    end
end