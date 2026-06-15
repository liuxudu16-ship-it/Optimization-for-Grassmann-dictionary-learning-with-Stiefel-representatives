
if ~exist('BATCH_MODE','var'), clc; clear; close all; end

cfg = experiment_config();
if cfg.runPreprocess && exist(cfg.preprocessScript, 'file')
    run(cfg.preprocessScript);
end

% 1数据文件路径
dataFile  = cfg.dataFile;
splitFile = cfg.splitFile;

% 2加载数据
fprintf('Loading: %s\n', dataFile);
S1 = load(dataFile);
fprintf('Loading: %s\n', splitFile);
S2 = load(splitFile);

X_all = S1.X_all;
y_all = S1.y_all(:);
p = S1.p;
splits = S2.splits;

[d, p2, nPoints] = size(X_all);
assert(p2 == p, 'p mismatch: p=%d but X_all has p=%d', p, p2);

nSplits = numel(splits);
nClasses = numel(unique(y_all));

fprintf('\nData loaded successfully!\n');
fprintf('  Total points:     %d\n', nPoints);
fprintf('  Feature dim (d):  %d\n', d);
fprintf('  Subspace dim (p): %d\n', p);
fprintf('  Number of classes: %d\n', nClasses);
fprintf('  Number of splits:  %d\n\n', nSplits);

lambda = cfg.lambda;
Tout = cfg.ToutBaseline;
Tin = cfg.TinBaseline;
atomsPerClass = cfg.atomsPerClass;
N = nClasses * atomsPerClass;

fprintf('==========================================\n');
fprintf('BASELINE: Controlled (Tout=0)\n');
fprintf('==========================================\n');
fprintf('Parameters (same as DL version, Tout=0):\n');
fprintf('  lambda = %.0e\n', lambda);
fprintf('  Tout   = %d\n', Tout);
fprintf('  Tin    = %d\n', Tin);
fprintf('  atomsPerClass = %d\n\n', atomsPerClass);

%5训练所有splits
fprintf('==========================================\n');
fprintf('Training %d Splits\n', nSplits);
fprintf('==========================================\n');
acc_list = zeros(nSplits,1);
time_list = zeros(nSplits,1);

for r = 1:nSplits
    fprintf('\n========== Split %d/%d ==========\n', r, nSplits);
    split_timer = tic;
    
    trIdx = splits(r).trIdx(:);
    teIdx = splits(r).teIdx(:);
    
    Xtr = X_all(:,:,trIdx);
    ytr = y_all(trIdx);
    mtr = numel(trIdx);
    
    Xte = X_all(:,:,teIdx);
    yte = y_all(teIdx);
    mte = numel(teIdx);
    
    fprintf('  Train: %d, Test: %d\n', mtr, mte);
    
    % 转cell
    Xtr_cell = cell(mtr,1);
    for i = 1:mtr
        Xtr_cell{i} = Xtr(:,:,i);
    end

    % 参考点
    B = grassmann_mean_proj(Xtr);

    % 统一初始化
    D0 = init_dictionary_shared(Xtr, ytr, atomsPerClass, ...
        cfg.initMode, cfg.nIterKmeans, cfg.initNoise, cfg.seedBase + r);

    % Tout=0: no dictionary learning
    opts.update_method = cfg.updateMethod;
    opts.gn_damping = cfg.gnDamping;
    [~, Y_star, LogD_star] = L_GDL_optim(Xtr_cell, D0, B, lambda, Tout, Tin, opts);

    % 测试集编码
    LogXte = cell(mte,1);
    for i = 1:mte
        LogXte{i} = grassmann_log(B, Xte(:,:,i));
    end
    Yte_code = L_GSC(LogXte, LogD_star, lambda);
    Ytr_code = double(Y_star);
    Yte_code = double(Yte_code);

    % 分类
    t = make_svm_template(cfg);
    svmMdl = fitcecoc(Ytr_code, ytr(:), ...
                      'Learners', t, ...
                      'Coding','onevsall', ...
                      'ClassNames', unique(ytr(:)));

    pred = predict(svmMdl, Yte_code);
    acc = mean(pred == yte(:));
    acc_list(r) = acc;
    
    time_list(r) = toc(split_timer);
    
    fprintf('  >>> Split %d accuracy: %.2f%% (%.2fs)\n', r, acc*100, time_list(r));
end

%% 结果汇总
fprintf('\n==========================================\n');
fprintf('FINAL RESULTS (Controlled Baseline)\n');
fprintf('==========================================\n');

mean_acc = mean(acc_list);
std_acc = std(acc_list);
min_acc = min(acc_list);
max_acc = max(acc_list);

fprintf('Accuracy Statistics:\n');
fprintf('  Mean:   %.2f%% (±%.2f%%)\n', mean_acc*100, std_acc*100);
fprintf('  Min:    %.2f%%\n', min_acc*100);
fprintf('  Max:    %.2f%%\n', max_acc*100);
fprintf('  Median: %.2f%%\n', median(acc_list)*100);

fprintf('\nTime Statistics:\n');
fprintf('  Total:   %.1fs (%.1f min)\n', sum(time_list), sum(time_list)/60);
fprintf('  Per split: %.1fs (%.1f min)\n', mean(time_list), mean(time_list)/60);

fprintf('\nAll splits:\n');
for r = 1:nSplits
    fprintf('  Split %2d: %.2f%% (%.1fs)\n', r, acc_list(r)*100, time_list(r));
end

fprintf('\n>>> 对比字典学习版本，观察提升幅度\n');
