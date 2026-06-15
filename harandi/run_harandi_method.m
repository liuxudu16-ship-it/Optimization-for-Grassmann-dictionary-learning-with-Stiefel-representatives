if ~exist('BATCH_MODE','var'), clc; clear; close all; end

rootDir = fileparts(fileparts(mfilename('fullpath')));
addpath(rootDir);
cfg = experiment_config();
if cfg.runPreprocess && exist(cfg.preprocessScript, 'file')
    run(cfg.preprocessScript);
end

%% 第1步：加载数据

fprintf('==========================================\n');
fprintf('COIL20 - Harandi Method (Fixed)\n');
fprintf('==========================================\n\n');

% 数据文件路径
dataFile  = cfg.dataFile;
splitFile = cfg.splitFile;

% 加载
fprintf('Loading data...\n');
S1 = load(dataFile);
S2 = load(splitFile);

X_all = S1.X_all;
y_all = S1.y_all(:);
p = S1.p;
splits = S2.splits;

[d, p2, nPoints] = size(X_all);
assert(p2 == p, 'p mismatch');

nSplits = numel(splits);
nClasses = numel(unique(y_all));

fprintf('Data loaded:\n');
fprintf('  Points: %d, d=%d, p=%d\n', nPoints, d, p);
fprintf('  Classes: %d, Splits: %d\n\n', nClasses, nSplits);

%% ========================================
%% 第2步：参数设置
%% ========================================

fprintf('==========================================\n');
fprintf('Parameter Settings\n');
fprintf('==========================================\n');

% Harandi方法的参数（与我们的方法一致）
dict_options.nIter = cfg.harandi.iter; % 字典学习迭代次数
dict_options.L = cfg.harandi.L;        % 稀疏度
dict_options.coding = cfg.harandi.coding;
dict_options.lambda = cfg.harandi.lambda;

% 字典原子数
atomsPerClass = cfg.atomsPerClass;
nAtoms = nClasses * atomsPerClass;

fprintf('Harandi method parameters:\n');
fprintf('  Dictionary atoms (nAtoms) = %d\n', nAtoms);
fprintf('  Dictionary learning iterations = %d\n', dict_options.nIter);
fprintf('  Sparsity level (L) = %d\n\n', dict_options.L);

%% ========================================
%% 第3步：训练所有splits
%% ========================================

fprintf('==========================================\n');
fprintf('Training %d Splits\n', nSplits);
fprintf('==========================================\n\n');

acc_list = zeros(nSplits,1);
time_list = zeros(nSplits,1);

for r = 1:nSplits
    fprintf('--- Split %d/%d ---\n', r, nSplits);
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
    
    %% 字典学习
    fprintf('  [1/4] Dictionary learning...\n');
    dict_timer = tic;
        D_star = grassmann_dictionary_learning(Xtr, nAtoms, dict_options);
        t_dict = toc(dict_timer);
        fprintf('    Dictionary learning: %.2fs\n', t_dict);

    %% 训练集编码
    fprintf('  [2/4] Encoding training set...\n');
    encode_timer = tic;
        alpha_tr = local_sparse_coding(Xtr, D_star, dict_options);
        t_encode = toc(encode_timer);
        fprintf('    Training encoding: %.2fs\n', t_encode);

    
    %% 测试集编码
    fprintf('  [3/4] Encoding test set...\n');
        alpha_te = local_sparse_coding(Xte, D_star, dict_options);

    %% 分类
    fprintf('  [4/4] Classification...\n');
    
    Ytr = full(double(alpha_tr'));
    Yte = full(double(alpha_te'));
 
        t = make_svm_template(cfg);
        
        svmMdl = fitcecoc(Ytr, ytr(:), ...
                          'Learners',t, ...
                          'Coding','onevsall', ...
                          'ClassNames', unique(ytr(:)));
        
        pred = predict(svmMdl, Yte);
        acc = mean(pred == yte(:));
        acc_list(r) = acc;

    time_list(r) = toc(split_timer);
    fprintf('  >>> Split %d accuracy: %.2f%% (%.2fs)\n\n', r, acc*100, time_list(r));
end

%% 第4步：结果汇总

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
fprintf('  Per split: %.1fs\n', mean(time_list));

fprintf('\nAll splits:\n');
for r = 1:nSplits
    fprintf('  Split %2d: %.2f%% (%.1fs)\n', r, acc_list(r)*100, time_list(r));
end

%% 第5步：保存结果

outDir = fullfile(cfg.rootDir, 'comparison_figures', 'harandi_results');
if ~exist(outDir,'dir')
    mkdir(outDir);
end

resultFile = fullfile(outDir, 'harandi_results.mat');
save(resultFile, 'acc_list', 'time_list', 'dict_options', 'nAtoms', ...
     'mean_acc', 'std_acc');
fprintf('Results saved: %s\n', resultFile);

summaryFile = fullfile(outDir, 'harandi_results_summary.txt');
fid = fopen(summaryFile, 'w');
fprintf(fid, 'COIL20 Harandi Method Results\n');
fprintf(fid, '=============================\n\n');
fprintf(fid, 'Method: Projection metric + %s\n\n', upper(dict_options.coding));
fprintf(fid, 'Parameters:\n');
fprintf(fid, '  Dictionary atoms = %d\n', nAtoms);
fprintf(fid, '  DL iterations = %d\n', dict_options.nIter);
fprintf(fid, '  Sparsity (L) = %d\n\n', dict_options.L);
fprintf(fid, 'Results:\n');
fprintf(fid, '  Mean accuracy: %.2f%% (±%.2f%%)\n', mean_acc*100, std_acc*100);
fprintf(fid, '  Min: %.2f%%, Max: %.2f%%\n', min_acc*100, max_acc*100);
fprintf(fid, '\nAll splits:\n');
for r = 1:nSplits
    fprintf(fid, '  Split %2d: %.2f%%\n', r, acc_list(r)*100);
end
fclose(fid);

%% 第6步：可视化
fig1 = figure('Color','w','Position',[100,100,1200,500]);

subplot(1,2,1);
bar(acc_list * 100, 'FaceColor', [0.8 0.4 0.2]);
hold on;
yline(mean_acc*100, 'r--', 'LineWidth', 2);
xlabel('Split', 'FontSize', 12);
ylabel('Accuracy (%)', 'FontSize', 12);
title('Harandi Method - Accuracy per Split', 'FontSize', 13);
grid on;
ylim([0, 100]);

subplot(1,2,2);
boxplot(acc_list * 100, 'Colors', [0.8 0.4 0.2]);
ylabel('Accuracy (%)', 'FontSize', 12);
title(sprintf('Accuracy Distribution\nMean=%.2f%%, Std=%.2f%%', ...
      mean_acc*100, std_acc*100), 'FontSize', 13);
grid on;
ylim([0, 100]);

saveas(fig1, fullfile(outDir, 'harandi_accuracy_results.png'));
savefig(fig1, fullfile(outDir, 'harandi_accuracy_results.fig'));

fprintf('✓ Visualizations saved\n');


fprintf('Results: %s\n', outDir);



%% OMP稀疏编码

function alpha = local_sparse_coding(X, dicX, dict_options)
% Sparse coding using OMP or LASSO
[~, p, ~] = size(dicX);
if isstruct(dict_options)
    if ~isfield(dict_options, 'L')
        dict_options.L = 10;
    end
    if ~isfield(dict_options, 'coding')
        dict_options.coding = 'omp';
    end
    if ~isfield(dict_options, 'lambda')
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

% 创建kernel矩阵
K_D = grassmann_proj_local(dicX);
K_XD = grassmann_proj_local(X, dicX);

% 准备向量计算
[KD_U, KD_D, ~] = svd(K_D);
D = diag(sqrt(diag(KD_D))) * KD_U' / sqrt(p);
D_Inv = KD_U * diag(1./sqrt(diag(KD_D)));
qX = D_Inv' * K_XD / sqrt(p);

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
    nAtoms = size(dicX, 3);
    nSamples = size(X, 3);
    alpha = zeros(nAtoms, nSamples);
end

end

%% ========================================
%% Grassmann投影距离（本地版本）
%% ========================================

function dist_p = grassmann_proj_local(SY1, SY2)
% 计算Grassmann投影距离
MIN_THRESH = 1e-6;

same_flag = false;
if nargin < 2
    SY2 = SY1;
    same_flag = true;
end

p = size(SY1, 2);

[~, ~, number_sets1] = size(SY1);
[~, ~, number_sets2] = size(SY2);

dist_p = zeros(number_sets2, number_sets1);

if same_flag
    for tmpC1 = 1:number_sets1
        Y1 = SY1(:,:,tmpC1);
        for tmpC2 = tmpC1:number_sets2
            tmpMatrix = Y1' * SY2(:,:,tmpC2);
            tmpProjection_Kernel_Val = sum(sum(tmpMatrix.^2));
            
            if tmpProjection_Kernel_Val < MIN_THRESH
                tmpProjection_Kernel_Val = 0;
            elseif tmpProjection_Kernel_Val > p
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
            
            if tmpProjection_Kernel_Val < MIN_THRESH
                tmpProjection_Kernel_Val = 0;
            elseif tmpProjection_Kernel_Val > p
                tmpProjection_Kernel_Val = p;
            end
            
            dist_p(tmpC2, tmpC1) = tmpProjection_Kernel_Val;
        end
    end
end

end
