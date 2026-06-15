function Dn = dict_update_gd_grassmann(X_all, D, alpha, step0, Tin)
% Dictionary update for the projection-matrix Grassmann model

    D = normalize_dictionary_format(D, X_all); 
    tol_grad = 1e-8;
    c_armijo = 1e-6;
    beta = 0.8;
    max_ls =10;

    [d, p, ~] = size(X_all);
    N = size(D, 3);

    Dn = D;
    J0 = cost_projmat(Dn, X_all, alpha, p);

    D_prev = Dn;
    Grad_prev = zeros(d, p, N);

    for t = 1:Tin
        [Grad, gnorm2] = grad_all_atoms(Dn, X_all, alpha);

        if ~isfinite(gnorm2) || gnorm2 < tol_grad
            break;
        end

        if t == 1
            step = step0;
        else
            if mod(t,2) == 0
                num = 0; den = 0;
                for r = 1:N
                    S = Dn(:,:,r) - D_prev(:,:,r);
                    Y = Grad(:,:,r) - Grad_prev(:,:,r) + ...
                        Dn(:,:,r) * (Dn(:,:,r)' * Grad_prev(:,:,r));
                    num = num + sum(S(:).^2);
                    den = den + abs(sum(S(:).*Y(:)));
                end
                step = num / (den + eps);
            else
                den = 0; yy = 0;
                for r = 1:N
                    S = Dn(:,:,r) - D_prev(:,:,r);
                    Y = Grad(:,:,r) - Grad_prev(:,:,r) + ...
                        Dn(:,:,r) * (Dn(:,:,r)' * Grad_prev(:,:,r));
                    den = den + abs(sum(S(:).*Y(:)));
                    yy  = yy  + sum(Y(:).^2);
                end
                step = den / (yy + eps);
            end
            step = max(min(step, 1e-1), 1e-12);
        end

        accepted = false;
        step_try = step;

        for ls = 1:max_ls
            D_try = retract_all(Dn, Grad, step_try);
            J_try = cost_projmat(D_try, X_all, alpha, p);

            if J_try <= J0 - c_armijo * step_try * gnorm2
                accepted = true;
                break;
            end
            step_try = step_try * beta;
        end

        if ~accepted
            break;
        end

        D_prev = Dn;
        Grad_prev = Grad;
        Dn = D_try;
        J0 = J_try;
    end
end

% ============================================================
function D = normalize_dictionary_format(D, X_all)
    d = size(X_all,1);
    p = size(X_all,2);

    if iscell(D)
        D = cat(3, D{:});
    end

    d0 = size(D,1);
    p0 = size(D,2);

    if d0 == d && p0 == p
        % ok
    elseif d0 == p && p0 == d
        Dt = zeros(d, p, size(D,3));
        for k = 1:size(D,3)
            Dt(:,:,k) = D(:,:,k)';
        end
        D = Dt;
    else
        error('Dictionary/data mismatch: X_all=[%d,%d,*], D=[%d,%d,*].', d,p,d0,p0);
    end

    for k = 1:size(D,3)
        [Q,~] = qr(D(:,:,k), 0);
        D(:,:,k) = Q;
    end
end

% ============================================================
function [Grad, gnorm2] = grad_all_atoms(D, X_all, alpha)
    [d, p, ~] = size(X_all);
    N = size(D, 3);

    Grad = zeros(d, p, N);
    gnorm2 = 0;

    for r = 1:N
        ar = alpha(r, :);
        idx = find(abs(ar) > 0);

        if isempty(idx)
            continue;
        end

        Dr = D(:, :, r);

        DjTDr = zeros(p, p, N);
        for j = 1:N
            DjTDr(:, :, j) = D(:, :, j)' * Dr;
        end

        term1 = zeros(d, p);
        term2 = zeros(d, p);

        for tt = 1:numel(idx)
            i = idx(tt);
            air = ar(i);

            Xi = X_all(:, :, i);
            term1 = term1 + air * (Xi * (Xi' * Dr));

            ai = alpha(:, i);
            nzj = find(abs(ai) > 0);
            nzj(nzj == r) = [];

            for k = 1:numel(nzj)
                j = nzj(k);
                term2 = term2 + (air * ai(j)) * (D(:, :, j) * DjTDr(:, :, j));
            end
        end

        gradE = -4 * (term1 - term2);
        gradR = gradE - Dr * (Dr' * gradE);

        Grad(:, :, r) = gradR;
        gnorm2 = gnorm2 + sum(gradR(:).^2);
    end
end

% ============================================================
function Dn = retract_all(D, Grad, step)
    N = size(D, 3);
    Dn = D;
    for r = 1:N
        Z = D(:, :, r) - step * Grad(:, :, r);
        Dn(:, :, r) = retraction_qr(Z);
    end
end

% ============================================================
function J = cost_projmat(D, X, alpha, p)
    m = size(X, 3);
    K_DD = grassmann_proj_kernel_local(D);
    K_DX = grassmann_proj_kernel_local(D, X)';
    J = p*m - 2*trace(alpha' * K_DX) + trace(alpha' * K_DD * alpha);
end

% ============================================================
function K = grassmann_proj_kernel_local(SY1, SY2)
    MIN_THRESH = 1e-6;

    if nargin < 2
        SY2 = SY1;
        same = true;
    else
        same = false;
    end

    p  = size(SY1, 2);
    n1 = size(SY1, 3);
    n2 = size(SY2, 3);

    K = zeros(n2, n1);

    if same
        for i = 1:n1
            Y1 = SY1(:,:,i);
            for j = i:n2
                v = sum(sum((Y1' * SY2(:,:,j)).^2));
                if v < MIN_THRESH, v = 0; end
                if v > p, v = p; end
                K(j,i) = v; K(i,j) = v;
            end
        end
    else
        for i = 1:n1
            Y1 = SY1(:,:,i);
            for j = 1:n2
                v = sum(sum((Y1' * SY2(:,:,j)).^2));
                if v < MIN_THRESH, v = 0; end
                if v > p, v = p; end
                K(j,i) = v;
            end
        end
    end
end

% ============================================================
function Q = retraction_qr(Z)
    [Q, R] = qr(Z, 0);
    sgn = sign(diag(R));
    sgn(sgn == 0) = 1;
    Q = Q * diag(sgn);
end