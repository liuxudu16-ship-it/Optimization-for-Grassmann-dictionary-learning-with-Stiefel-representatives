function [Grad, FuGrad, gnorm2, should_break] = GD_grad_fixed(D, G, B, c_cache, t, J0, tol_grad)
N = numel(D);
Grad = cell(1, N);
FuGrad = cell(1, N);
norm_local = zeros(N, 1);

% 并行计算每个原子的梯度
parfor j = 1:N
    Dj = D{j};
    cj = c_cache{j};
    Grad_j = adlog(G{j}, B, Dj, cj.Q1, cj.S1, cj.R1,cj.Q2, cj.S2, cj.Sigma, cj.R2);
    
    %投影到切空间（黎曼梯度）
    Grad_j = Grad_j - Dj*(Dj'*Grad_j);
    
    grad_norm_j = norm(Grad_j, 'fro');
    Grad{j} = Grad_j;
    FuGrad{j} = -Grad_j; 
    norm_local(j) = grad_norm_j^2;
end

% 计算总梯度范数
gnorm2 = sum(norm_local);
grad_norm = sqrt(gnorm2);

% 检查收敛
should_break = false;

% 输出信息
if isnan(grad_norm) || isinf(grad_norm)
    fprintf('  Iter %d: J=%.6e, ||grad||=NaN/Inf\n', t, J0);
    should_break = true;
else
    if mod(t, 5) == 0 || t == 1  % 每5轮输出一次，减少输出量
        fprintf('  [%3d] J=%.6e, ||grad||=%.3e\n', t, J0, grad_norm);
    end
    
    if grad_norm < tol_grad
        fprintf('  Gradient norm %.3e below tolerance %.3e\n', grad_norm, tol_grad);
        should_break = true;
    end
end
end

