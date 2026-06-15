function t = make_svm_template(cfg)
%make_svm_template Build a shared SVM template from config.

args = { ...
    'KernelFunction', cfg.svm.kernel, ...
    'Standardize', cfg.svm.standardize, ...
    'BoxConstraint', cfg.svm.box ...
};

if isfield(cfg.svm, 'kernelScale') && ~strcmpi(cfg.svm.kernel, 'linear')
    args = [args, {'KernelScale', cfg.svm.kernelScale}];
end

t = templateSVM(args{:});
end
