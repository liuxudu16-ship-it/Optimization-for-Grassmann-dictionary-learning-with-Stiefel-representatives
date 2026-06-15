function D0 = init_dictionary_shared(Xtr, ytr, atomsPerClass, initMode, nIterKmeans, initNoise, rngSeed)
%init_dictionary_shared Initialize dictionary atoms consistently.

if nargin < 7
    rngSeed = [];
end
if ~isempty(rngSeed)
    rng(rngSeed, 'twister');
end

classes = unique(ytr(:))';
nClasses = numel(classes);
N = nClasses * atomsPerClass;
D0 = cell(N, 1);

jj = 1;

switch lower(initMode)
    case 'kmeans'
        for ci = 1:nClasses
            c = classes(ci);
            idc = find(ytr == c);
            if isempty(idc)
                error('Class %d has no training samples.', c);
            end
            Xc = Xtr(:,:,idc);
            K = min(atomsPerClass, size(Xc, 3));
            centers = kmeans_projection(Xc, K, nIterKmeans, 0);
            for k = 1:K
                D0{jj} = centers(:,:,k);
                jj = jj + 1;
            end
        end
    case 'random_class'
        for ci = 1:nClasses
            c = classes(ci);
            idc = find(ytr == c);
            if isempty(idc)
                error('Class %d has no training samples.', c);
            end
            K = min(atomsPerClass, numel(idc));
            pick = idc(randperm(numel(idc), K));
            for k = 1:K
                Xi = Xtr(:,:,pick(k));
                if initNoise > 0
                    Xi = retraction_qr(Xi + initNoise * randn(size(Xi)));
                end
                D0{jj} = Xi;
                jj = jj + 1;
            end
        end
    case 'random_all'
        allIdx = 1:size(Xtr, 3);
        pick = allIdx(randperm(numel(allIdx), min(N, numel(allIdx))));
        for k = 1:numel(pick)
            Xi = Xtr(:,:,pick(k));
            if initNoise > 0
                Xi = retraction_qr(Xi + initNoise * randn(size(Xi)));
            end
            D0{jj} = Xi;
            jj = jj + 1;
        end
    case 'class_mean'
        [d, p, ~] = size(Xtr);
        for ci = 1:nClasses
            c = classes(ci);
            idc = find(ytr == c);
            if isempty(idc)
                error('Class %d has no training samples.', c);
            end
            Xc = Xtr(:,:,idc);
            Pc = zeros(d, d);
            for ii = 1:size(Xc, 3)
                Pc = Pc + Xc(:,:,ii) * Xc(:,:,ii)';
            end
            Pc = Pc / size(Xc, 3);
            [V, Dv] = eig((Pc + Pc') / 2);
            [~, ord] = sort(diag(Dv), 'descend');
            Umean = V(:, ord(1:p));
            Umean = retraction_qr(Umean);
            D0{jj} = Umean;
            jj = jj + 1;
        end
    otherwise
        error('Unknown initMode: %s', initMode);
end

% Fill remaining atoms if needed
if jj <= N
    allIdx = 1:size(Xtr, 3);
    remain = N - (jj - 1);
    pick2 = allIdx(randperm(numel(allIdx), remain));
    for rr = 1:remain
        Xi = Xtr(:,:,pick2(rr));
        if initNoise > 0
            Xi = retraction_qr(Xi + initNoise * randn(size(Xi)));
        end
        D0{jj} = Xi;
        jj = jj + 1;
    end
end
end
