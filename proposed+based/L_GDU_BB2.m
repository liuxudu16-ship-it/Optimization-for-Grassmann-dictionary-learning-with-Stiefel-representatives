function [D, LogD, c_cache] = L_GDU_BB2(LogX, LogD, D, c_cache, Y_code, Tin, B, N)
%% 初始化
tol_grad = 1e-6;

%% 预分配
D_next = cell(1, N);
LogD_next = cell(1, N);
c_cache_next = cell(1, N);

D_prev = D;
Grad_prev = cell(1, N);
for j = 1:N
    Grad_prev{j} = zeros(size(D{j}));
end

%% 初始 LogD / cache
parfor j = 1:N
    [LogD{j}, c_cache{j}] = grassmann_log(B, D{j});
end

% 初始代价与梯度缓存G
[J0, ~, ~, G] = costfun(LogX, LogD, Y_code);

%% 迭代优化
for t = 1:Tin

    stop_update = false;

    % 用最新 D 重算 LogD/cache，并更新 J0/G（t>1）
    if t > 1
        parfor j = 1:N
            [LogD{j}, c_cache{j}] = grassmann_log(B, D{j});
        end
        [J0, ~, ~, G] = costfun(LogX, LogD, Y_code);
    end

    % 计算梯度 + 收敛判断
    [Grad, ~, gnorm2, should_break] = GD_grad_fixed(D, G, B, c_cache, tol_grad);
    if should_break
        break;
    end

    %% BB 步长
    if t == 1
        step = 1e-4;
    else
        num = 0; den = 0; grad = 0;
        if mod(t, 2) == 0
            % BB1: (s^T s) / (s^T y)
            for j = 1:N
                S = D{j} - D_prev{j};
                Y = Grad{j} - Grad_prev{j} + D{j} * (D{j}' * Grad_prev{j});
                num = num + sum(S(:).^2);
                den = den + abs(sum(sum(S .* Y)));
            end
            step = num / (den + eps);
        else
            % BB2: (s^T y) / (y^T y)
            for j = 1:N
                S = D{j} - D_prev{j};
                Y = Grad{j} - Grad_prev{j} + D{j} * (D{j}' * Grad_prev{j});
                grad = grad + sum(Y(:).^2);
                den  = den + abs(sum(sum(S .* Y)));
            end
            step = den / (grad + eps);
        end

        step = max(min(step, 1e-1), 1e-10);
    end

    %% Armijo 线搜索
    c_armijo = 1e-6;
    beta=0.8;
    max_ls =20;
    accepted = false;

    for ls = 1:max_ls
        parfor j = 1:N
            D_next{j} = retraction_qr(D{j} - step * Grad{j});
            [LogD_next{j}, c_cache_next{j}] = grassmann_log(B, D_next{j});
        end

        [J_next, ~, ~, G_next] = costfun(LogX, LogD_next, Y_code);

        if J_next <= J0 - c_armijo * step * gnorm2
            accepted = true;
            break;
        else
            step = step * beta;
        end
    end

    %% 应用更新 / 失败则退出字典更新
    if accepted
        D_prev = D;
        Grad_prev = Grad;

        c_cache = c_cache_next;
        D = D_next;
        LogD = LogD_next;

        J0 = J_next;
        G = G_next;
    else
        stop_update = true;
    end

    if stop_update
        break;
    end
end

end

function [Grad, FuGrad, gnorm2, should_break] = GD_grad_fixed(D, G, B, c_cache, tol_grad)
N = numel(D);
Grad = cell(1, N);
FuGrad = cell(1, N);
norm_local = zeros(N, 1);

parfor j = 1:N
    Dj = D{j};
    cj = c_cache{j};

    Grad_j = adlog(G{j}, B, Dj, ...
                   cj.Q1, cj.S1, cj.R1, ...
                   cj.Q2, cj.S2, cj.Sigma, cj.R2);

    % 投影到切空间（黎曼梯度）
    Grad_j = Grad_j - Dj*(Dj'*Grad_j);

    Grad{j} = Grad_j;
    FuGrad{j} = -Grad_j;
    norm_local(j)=norm(Grad_j, 'fro')^2;
end

gnorm2 = sum(norm_local);
grad_norm = sqrt(gnorm2);

% 收敛/异常判断
should_break = false;
if isnan(grad_norm) || isinf(grad_norm)
    should_break = true;
    return;
end
if grad_norm < tol_grad
    should_break = true;
    return;
end

end
