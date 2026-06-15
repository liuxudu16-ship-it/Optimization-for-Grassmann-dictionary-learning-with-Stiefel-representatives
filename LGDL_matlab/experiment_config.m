function cfg = experiment_config()
%experiment_config Shared experiment settings.

cfg.rootDir = fileparts(mfilename('fullpath'));

% Dataset selection: 'data' (new ORL), 'data2' (legacy ORL), or 'coil20'
cfg.dataset = 'gesture';
env_dataset = getenv('CFG_DATASET');
if ~isempty(env_dataset)
    cfg.dataset = env_dataset;
end

switch lower(cfg.dataset)
    case 'data'
        cfg.dataDir = fullfile(cfg.rootDir,'data\data2\data_out');
        cfg.dataFile = fullfile(cfg.dataDir,'orl_grassmann.mat');
        cfg.splitFile = fullfile(cfg.dataDir, 'orl_splits10.mat');
        cfg.preprocessScript = fullfile(cfg.rootDir, 'data', 'data2','preprocess_orl.m');
    case 'coil20'
        cfg.dataDir = fullfile(cfg.rootDir, 'coil20_5', 'data_out');
        cfg.dataFile = fullfile(cfg.dataDir, 'coil20_grassmann.mat');
        cfg.splitFile = fullfile(cfg.dataDir, 'coil20_splits10.mat');
        cfg.preprocessScript = fullfile(cfg.rootDir, 'preprocess_coil20_grassmann_optimal.m');
    case 'ballet'
        cfg.dataDir = fullfile(cfg.rootDir, 'ballet', 'data_out');
        cfg.dataFile = fullfile(cfg.dataDir, 'ballet_grassmann.mat');
        cfg.splitFile = fullfile(cfg.dataDir, 'ballet_splits10.mat');
        cfg.preprocessScript = fullfile(cfg.rootDir, 'preprocess_coil20_grassmann_optimal.m');   
    case 'gesture'
        cfg.dataDir = fullfile(cfg.rootDir, 'gestureData', 'data_out');
        cfg.dataFile = fullfile(cfg.dataDir, 'msrgesture3d_grassmann.mat');
        cfg.splitFile = fullfile(cfg.dataDir, 'msrgesture3d_splits10.mat');
        cfg.preprocessScript = fullfile(cfg.rootDir, 'preprocess_msrgesture3d_grassmann.m');
    otherwise
        error('Unknown dataset: %s', cfg.dataset);
end

cfg.runPreprocess =false;

% Dictionary learning parameters
cfg.lambda =1e-3;
cfg.Tout =0;
cfg.Tin =0;
cfg.atomsPerClass =5;

% Baseline (no dictionary learning)
cfg.ToutBaseline = 0;
cfg.TinBaseline = 0;

% Initialization
cfg.initMode ='random_all'; % kmeans || random_all 
cfg.nIterKmeans =5;
cfg.initNoise =0;
cfg.seedBase = 1000;
cfg.maxSplits = 1;

% Update method: 'bb2' (gradient) or 'gn' 
cfg.updateMethod = 'bb2';
cfg.gnDamping = 1e-6;

% SVM settings
cfg.svm.kernel = 'rbf';
cfg.svm.kernelScale = 'auto';
cfg.svm.box =10;
cfg.svm.standardize = true;

% Harandi settings
cfg.harandi.iter =0;
cfg.harandi.L=0;
cfg.harandi.coding ='omp';
cfg.harandi.lambda = cfg.lambda;
cfg.harandi.script = fullfile(cfg.rootDir, '对比', 'run_harandi_method.m');

% Baseline script for run_all_test/one-click
cfg.baselineScript = fullfile(cfg.rootDir, 'run_baseline_no_dict.m');
cfg.baselineName ='No dictionary learning (Tout=0)';
cfg.baselineNoise = 0;

% Optional environment overrides for quick experiments
cfg = apply_env_overrides(cfg);
end

function cfg = apply_env_overrides(cfg)
%apply_env_overrides Allow quick tuning via environment variables.

cfg = apply_override(cfg, 'lambda', 'CFG_LAMBDA',@str2double);
cfg = apply_override(cfg, 'Tout', 'CFG_TOUT',@str2double);
cfg = apply_override(cfg, 'Tin', 'CFG_TIN',@str2double);
cfg = apply_override(cfg, 'atomsPerClass', 'CFG_ATOMS_PER_CLASS',@str2double);
cfg = apply_override(cfg, 'initMode', 'CFG_INIT_MODE',@char);
cfg = apply_override(cfg, 'maxSplits', 'CFG_MAX_SPLITS',@str2double);
cfg = apply_override(cfg, 'updateMethod', 'CFG_UPDATE_METHOD',@char);

cfg.harandi = apply_override(cfg.harandi, 'iter', 'CFG_HARANDI_ITER', @str2double);
cfg.harandi = apply_override(cfg.harandi, 'L', 'CFG_HARANDI_L', @str2double);
cfg.harandi = apply_override(cfg.harandi, 'coding', 'CFG_HARANDI_CODING', @char);
cfg.harandi = apply_override(cfg.harandi, 'lambda', 'CFG_HARANDI_LAMBDA', @str2double);

cfg.svm = apply_override(cfg.svm, 'kernel', 'CFG_SVM_KERNEL', @char);
cfg.svm = apply_override(cfg.svm, 'kernelScale', 'CFG_SVM_KERNEL_SCALE', @parse_numeric_or_string);
cfg.svm = apply_override(cfg.svm, 'box', 'CFG_SVM_BOX', @str2double);
cfg.svm = apply_override(cfg.svm, 'standardize', 'CFG_SVM_STANDARDIZE', @parse_bool);
end

function s = apply_override(s, field, env_name, conv)
val = getenv(env_name);
if isempty(val)
    return;
end
parsed = conv(val);
if ~ischar(parsed) && isnan(parsed)
    return;
end
s.(field) = parsed;
end

function out = parse_numeric_or_string(val)
num = str2double(val);
if ~isnan(num)
    out = num;
else
    out = char(val);
end
end

function out = parse_bool(val)
val = lower(strtrim(val));
if any(strcmp(val, {'1','true','yes'}))
    out = true;
elseif any(strcmp(val, {'0','false','no'}))
    out = false;
else
    out = true;
end
end
