clc; clear; close all;

%% ========== 配置 ==========
cfg = experiment_config();
rng(cfg.seedBase, 'twister');

%% ========== SPAMS 路径 ==========
spams_build = fullfile(cfg.rootDir, 'third_party', 'spams-matlab-v2.6', 'build');
if exist(spams_build, 'dir')
    addpath(spams_build, '-begin');
end

rehash;

%% ========== 手动指定一个实验文件夹 ==========
expDir = 'D:\project1\坚持就是胜利 4\坚持就是胜利 2\data3\data\data1_2(2)';

dataFile  = fullfile(expDir, 'data_out', 'orl_grassmann.mat');
splitFile = fullfile(expDir, 'data_out', 'orl_splits10.mat');

% 这里必须是之前保存过 D0_all 和 B_all 的结果文件
% 如果你的文件在 comparison_figures 里面，就改成：
% initFile = fullfile(expDir, 'comparison_figures', 'experiment_results_for_plot.mat');
initFile  = fullfile(expDir, 'experiment_results_for_plot.mat');

%% ========== 加载 ==========
fprintf('Loading data...\n');

S1 = load(dataFile);
S2 = load(splitFile);
S3 = load(initFile);

X_all  = S1.X_all;
y_all  = S1.y_all(:);
p      = S1.p;
splits = S2.splits;

plotData = S3.plotData;

if ~isfield(plotData, 'D0_all')
    error('initFile 中的 plotData 没有 D0_all。请先运行保存初始化字典的主实验代码。');
end

if ~isfield(plotData, 'B_all')
    warning('initFile 中没有 B_all，后续将重新根据训练集计算 B。');
    plotData.B_all = cell(size(plotData.D0_all));
end

D0_all = plotData.D0_all;
B_all  = plotData.B_all;

[d, ~, nPoints] = size(X_all);
nSplits  = numel(splits);
nClasses = numel(unique(y_all));

if isfield(cfg, 'maxSplits') && ~isempty(cfg.maxSplits)
    nSplits = min(nSplits, cfg.maxSplits);
    splits = splits(1:nSplits);
end

fprintf('Data: %d points, %d classes, p=%d\n', nPoints, nClasses, p);
fprintf('Using old D0/B from: %s\n\n', initFile);

%% ========== LGDL 参数 ==========
lambda = cfg.lambda;
Tin    = cfg.Tin;
Tout   = cfg.Tout;

opts.update_method = cfg.updateMethod;   % 'bb2' or 'gn'
opts.gn_damping    = cfg.gnDamping;

fprintf('==========================================\n');
fprintf('LGDL ONLY RUN\n');
fprintf('==========================================\n');
fprintf('lambda       = %.3e\n', lambda);
fprintf('Tout         = %d\n', Tout);
fprintf('Tin          = %d\n', Tin);
fprintf('updateMethod = %s\n', opts.update_method);
fprintf('gnDamping    = %.3e\n', opts.gn_damping);
fprintf('==========================================\n\n');

%% ========== 结果变量 ==========
acc_lgdl  = zeros(nSplits, 1);
time_lgdl = zeros(nSplits, 1);

cost_lgdl_all = cell(nSplits, 1);

D0_used_all   = cell(nSplits, 1);
B_ref_all     = cell(nSplits, 1);
D_lgdl_all    = cell(nSplits, 1);
LogD_lgdl_all = cell(nSplits, 1);
Y_lgdl_all    = cell(nSplits, 1);

%% ========== 主循环 ==========
for r = 1:nSplits
    fprintf('\n========== Split %d/%d ==========\n', r, nSplits);

    trIdx = get_split_field(splits(r), {'trIdx','tridx','trldx','trLdx'});
    teIdx = get_split_field(splits(r), {'teIdx','teidx','teldx','teLdx'});

    trIdx = trIdx(:);
    teIdx = teIdx(:);

    Xtr = X_all(:,:,trIdx);
    ytr = y_all(trIdx);

    Xte = X_all(:,:,teIdx);
    yte = y_all(teIdx);

    mtr = numel(trIdx);

    %% ----- 训练集转成 cell，供 L_GDL_optim_with_history 使用 -----
    Xtr_cell = cell(mtr, 1);
    for i = 1:mtr
        Xtr_cell{i} = Xtr(:,:,i);
    end

    %% ----- 读取同一个初始化字典 D0 -----
    D0_raw = D0_all{r};
    D0 = convert_init_dictionary_to_cell(D0_raw);
    D0 = align_D0_cell_to_X(D0, Xtr);

    D0_used_all{r} = D0;

    %% ----- 读取同一个参考点 B；如果没有，则重新计算 -----
    if ~isempty(B_all) && r <= numel(B_all) && ~isempty(B_all{r})
        B = align_basis_to_X(B_all{r}, Xtr);
    else
        fprintf('B_all{%d} is empty. Recomputing B by grassmann_mean_proj(Xtr)...\n', r);
        B = grassmann_mean_proj(Xtr);
        B = align_basis_to_X(B, Xtr);
    end

    B_ref_all{r} = B;

    fprintf('size(Xtr) = [%d, %d, %d]\n', size(Xtr,1), size(Xtr,2), size(Xtr,3));
    fprintf('numel(D0) = %d\n', numel(D0));
    fprintf('size(D0{1}) = [%d, %d]\n', size(D0{1},1), size(D0{1},2));
    fprintf('size(B) = [%d, %d]\n', size(B,1), size(B,2));

    %% ----- LGDL / Proposed Method -----
    fprintf('Running LGDL / Proposed Method...\n');

    t = tic;

    [D_lgdl, Y_lgdl, LogD_lgdl, cost_history] = ...
        L_GDL_optim_with_history(Xtr_cell, D0, B, lambda, Tout, Tin, opts);

    time_lgdl(r) = toc(t);

    cost_lgdl_all{r} = cost_history;
    D_lgdl_all{r}    = D_lgdl;
    LogD_lgdl_all{r} = LogD_lgdl;
    Y_lgdl_all{r}    = Y_lgdl;

    %% ----- 测试集编码 + SVM 分类 -----
    acc_lgdl(r) = eval_accuracy_lgdl(Xte, yte, ytr, B, Y_lgdl, LogD_lgdl, lambda, cfg);

    fprintf('LGDL acc = %.2f%%, time = %.2fs\n', ...
        acc_lgdl(r) * 100, time_lgdl(r));
end

%% ========== 汇总 ==========
mean_acc = mean(acc_lgdl) * 100;
std_acc  = std(acc_lgdl) * 100;
total_t  = sum(time_lgdl);
mean_t   = mean(time_lgdl);
std_t    = std(time_lgdl);

fprintf('\n==========================================\n');
fprintf('LGDL FINAL RESULTS\n');
fprintf('==========================================\n');
fprintf('Accuracy: %.2f%% (±%.2f%%)\n', mean_acc, std_acc);
fprintf('Total time: %.2fs\n', total_t);
fprintf('Mean time per split: %.2fs\n', mean_t);
fprintf('Std time per split: %.2fs\n', std_t);

%% ========== 平均 loss 曲线 ==========
[cost_history_lgdl_mean, cost_history_lgdl_std] = average_cost_curves(cost_lgdl_all);

%% ========== 保存结果 ==========
resultDir = fullfile(expDir, 'proposed_figures');
if ~exist(resultDir, 'dir')
    mkdir(resultDir);
end

resultFile = fullfile(resultDir, 'proposed_results.mat');

save(resultFile, ...
    'acc_lgdl', 'time_lgdl', 'cost_lgdl_all', ...
    'cost_history_lgdl_mean', 'cost_history_lgdl_std', ...
    'mean_acc', 'std_acc', 'total_t', 'mean_t', 'std_t', ...
    'D0_used_all', 'B_ref_all', ...
    'D_lgdl_all', 'LogD_lgdl_all', 'Y_lgdl_all', ...
    'lambda', 'Tout', 'Tin', 'opts', ...
    'dataFile', 'splitFile', 'initFile', '-v7.3');

fprintf('\nSaved to %s\n', resultFile);

%% ========== 图1：准确率箱线图 ==========
fig1 = figure('Visible','off','Color','w','Position',[100 100 700 500]);
boxplot(acc_lgdl * 100, 'Labels', {'LGDL'});
grid on;
ylabel('Accuracy (%)');
title('Accuracy Distribution of LGDL');
savefig(fig1, fullfile(resultDir, 'accuracy_boxplot_lgdl.fig'));
exportgraphics(fig1, fullfile(resultDir, 'accuracy_boxplot_lgdl.png'), 'Resolution', 300);
close(fig1);

%% ========== 图2：每个 split 的准确率 ==========
fig2 = figure('Visible','off','Color','w','Position',[100 100 900 500]);
bar(acc_lgdl * 100);
grid on;
xlabel('Split');
ylabel('Accuracy (%)');
title('Accuracy per Split - LGDL');
savefig(fig2, fullfile(resultDir, 'accuracy_per_split_lgdl.fig'));
exportgraphics(fig2, fullfile(resultDir, 'accuracy_per_split_lgdl.png'), 'Resolution', 300);
close(fig2);

%% ========== 图3：平均 loss 曲线 ==========
if ~isempty(cost_history_lgdl_mean)
    fig3 = figure('Visible','off','Color','w','Position',[100 100 900 500]);
    x = 0:(numel(cost_history_lgdl_mean)-1);

    plot(x, cost_history_lgdl_mean, '-o', 'LineWidth', 1.8, 'MarkerSize', 6);
    grid on;
    xlabel('Outer Iteration');
    ylabel('Loss Value');
    title('Loss Curve of LGDL (Avg over Splits)');

    savefig(fig3, fullfile(resultDir, 'loss_curve_lgdl.fig'));
    exportgraphics(fig3, fullfile(resultDir, 'loss_curve_lgdl.png'), 'Resolution', 300);
    close(fig3);
end

%% ========== 图4：每个 split 的运行时间 ==========
fig4 = figure('Visible','off','Color','w','Position',[100 100 900 500]);
bar(time_lgdl);
grid on;
xlabel('Split');
ylabel('Time (s)');
title('Runtime per Split - LGDL');
savefig(fig4, fullfile(resultDir, 'runtime_per_split_lgdl.fig'));
exportgraphics(fig4, fullfile(resultDir, 'runtime_per_split_lgdl.png'), 'Resolution', 300);
close(fig4);

%% ========== 图5：结果表 ==========
fig5 = figure('Visible','off','Color','w','Position',[100 100 900 220]);

tableData = { ...
    'LGDL', ...
    sprintf('%.2f%%', mean_acc), ...
    sprintf('±%.2f%%', std_acc), ...
    sprintf('%.1fs', total_t)};

uitable('Parent', fig5, ...
    'Data', tableData, ...
    'ColumnName', {'Method', 'Accuracy', 'Std', 'Total Time'}, ...
    'RowName', [], ...
    'Units', 'normalized', ...
    'Position', [0 0 1 1], ...
    'FontSize', 12);

drawnow;
savefig(fig5, fullfile(resultDir, 'results_table_lgdl.fig'));
exportgraphics(fig5, fullfile(resultDir, 'results_table_lgdl.png'), 'Resolution', 300);
close(fig5);

fprintf('\nSaved all figures to:\n%s\n', resultDir);

%% =========================================================
%% 辅助函数
%% =========================================================

function acc = eval_accuracy_lgdl(Xte, yte, ytr, B, Ytr_code, LogD, lambda, cfg)
    mte = size(Xte, 3);

    LogXte = cell(mte, 1);
    for i = 1:mte
        LogXte{i} = grassmann_log(B, Xte(:,:,i));
    end

    Yte_code = L_GSC(LogXte, LogD, lambda);

    Ytr_code = double(Ytr_code);
    Yte_code = double(Yte_code);

    t = make_svm_template(cfg);

    svmMdl = fitcecoc(Ytr_code, ytr(:), ...
        'Learners', t, ...
        'Coding', 'onevsall', ...
        'ClassNames', unique(ytr(:)));

    pred = predict(svmMdl, Yte_code);
    acc = mean(pred == yte(:));
end

function [mean_curve, std_curve] = average_cost_curves(cost_cell)
    n = numel(cost_cell);
    lens = zeros(n,1);

    for i = 1:n
        if isempty(cost_cell{i})
            lens(i) = 0;
        else
            lens(i) = numel(cost_cell{i});
        end
    end

    Tmax = max(lens);
    if Tmax == 0
        mean_curve = [];
        std_curve = [];
        return;
    end

    M = nan(n, Tmax);
    for i = 1:n
        ci = cost_cell{i};
        if isempty(ci)
            continue;
        end

        ci = ci(:)';
        M(i, 1:numel(ci)) = ci;
    end

    mean_curve = nanmean(M, 1);
    std_curve  = nanstd(M, 0, 1);
end

function val = get_split_field(s, names)
    for k = 1:numel(names)
        if isfield(s, names{k})
            val = s.(names{k});
            return;
        end
    end

    error('split field not found. Please check split struct field names.');
end

function D0_cell = convert_init_dictionary_to_cell(D0_in)
    % 目标格式：N×1 cell，每个元素为 d×p

    if iscell(D0_in)
        D0_cell = D0_in(:);
        return;
    end

    if isnumeric(D0_in) && ndims(D0_in) == 3
        N = size(D0_in, 3);
        D0_cell = cell(N, 1);
        for k = 1:N
            D0_cell{k} = D0_in(:,:,k);
        end
        return;
    end

    error('Cannot recognize D0 format. D0 should be cell or d×p×N numeric array.');
end

function D0_cell = align_D0_cell_to_X(D0_cell, Xtr)
    d = size(Xtr, 1);
    p = size(Xtr, 2);

    for k = 1:numel(D0_cell)
        A = D0_cell{k};

        if size(A,1) == d && size(A,2) == p
            % ok
        elseif size(A,1) == p && size(A,2) == d
            A = A';
        else
            error('D0{%d} 与当前 Xtr 维度不匹配。size(D0{%d})=[%d,%d], expected [%d,%d].', ...
                k, k, size(A,1), size(A,2), d, p);
        end

        [Q, ~] = qr(A, 0);
        D0_cell{k} = Q;
    end
end

function B = align_basis_to_X(B_in, Xtr)
    d = size(Xtr, 1);
    p = size(Xtr, 2);

    B = B_in;

    if size(B,1) == d && size(B,2) == p
        % ok
    elseif size(B,1) == p && size(B,2) == d
        B = B';
    else
        error('B 与当前 Xtr 维度不匹配。size(B)=[%d,%d], expected [%d,%d].', ...
            size(B,1), size(B,2), d, p);
    end

    [Q, ~] = qr(B, 0);
    B = Q;
end