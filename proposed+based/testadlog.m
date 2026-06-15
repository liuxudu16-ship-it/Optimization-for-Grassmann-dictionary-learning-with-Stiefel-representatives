
    d = 6; p = 3;
    % --- random orthonormal U (base) and Y (point) ---
    U= orth(randn(d,p));
    Y= orth(randn(d,p));
    % --- forward log to get cache ---
    [~, cache] = grassmann_log(U, Y);
    Q1=cache.Q1; S1=cache.S1; R1=cache.R1;
    Q2=cache.Q2; S2=cache.S2; R2=cache.R2; Sigma=cache.Sigma;
    % --- random tangent direction at Y ---
    Ydot = randn(d,p);
    Ydot = Ydot - Y*(Y'*Ydot);   % tangent projection
    % --- random G (same size as Delta) ---
    G = randn(d,p);
    % --- apply differential and adjoint ---
    dlogY = dlog(Y,U,Q1,S1,R1,Q2,S2,R2,Sigma,Ydot);
    adlogG = adlog(G,U,Y,Q1,S1,R1,Q2,S2,Sigma,R2);
    % --- Euclidean / Frobenius inner products ---
    ip_left  = trace(dlogY' * G);     % < dlog[Y_dot], G >
    ip_right = trace(Ydot' * adlogG); % < Y_dot, dlog^*[G] >
    relerr = abs(ip_left - ip_right) / max([1, abs(ip_left), abs(ip_right)]);
    % --- readable output ---
    fprintf("====== Euclidean inner-product adjointness check ======\n");
    fprintf("< dlog_U(Y)[Y_dot] , G >    = %.16e\n", ip_left);
    fprintf("< Y_dot , dlog_U(Y)^*[G] > = %.16e\n", ip_right);
    fprintf("relative error             = %.3e\n", relerr);
    fprintf("=======================================================\n");

