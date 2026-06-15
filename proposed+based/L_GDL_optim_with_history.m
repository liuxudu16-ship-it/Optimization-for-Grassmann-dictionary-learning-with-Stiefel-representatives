function [D_star, Y_star, LogD_star, cost_history] = L_GDL_optim_with_history(X, D0, B, lambda, Tout, Tin, opts)
m = numel(X);
N = numel(D0);

%% Log-map
fprintf('Computing log-maps for training data...\n');
LogX = cell(1, m);
parfor i = 1:m
    LogX{i} = grassmann_log(B, X{i});
end

%% 初始化字典 LogD
fprintf('Initializing dictionary...\n');
LogD = cell(1, N);
c_cache = cell(1, N);
parfor j = 1:N
    [LogD{j}, c_cache{j}] = grassmann_log(B, D0{j});
end

D = D0;

%% 预分配
cost_history = zeros(1, Tout + 1);

%% Initial cost
Y_init = L_GSC(LogX, LogD, lambda);
[J_init, ~, ~, ~] = costfun(LogX, LogD, Y_init);
cost_history(1) = J_init;
fprintf('Initial cost -->%.3f\n', J_init);

%% Tout=0：只做编码，不学习
if Tout == 0
    D_star = D;
    LogD_star = LogD;
    Y_star = Y_init;
    fprintf('-------\n');
    return;
end

for t = 1:Tout
    % 1) Sparse coding
    Y_code = L_GSC(LogX, LogD, lambda);
    
    % 2) Dictionary update
    switch lower(opts.update_method)
        case 'gn'
            [D, LogD, c_cache] = L_GDU_GN(LogX, D, Y_code, B, opts.gn_damping);
        otherwise
            [D, LogD, c_cache] = L_GDU_BB2(LogX, LogD, D, c_cache, Y_code, Tin, B, N);
    end

    % 3) Cost after update
    [J_curr, ~, ~, ~] = costfun(LogX, LogD, Y_code);
    cost_history(t+1) = J_curr;

    fprintf('Iter#%d: cost -->%.3f\n', t, J_curr);
end
fprintf('-------\n');

%% 输出
D_star = D;
LogD_star = LogD;
Y_star = L_GSC(LogX, LogD, lambda);

end
