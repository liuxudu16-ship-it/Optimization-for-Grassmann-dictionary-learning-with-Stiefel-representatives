function xi_dot = differential(Y, U, Q1, S1, R1, Q2, S2, R2, Sigma, Y_dot)
    % --- diag vectors ---
    s1  = diag(S1);    S1m = diag(s1);
    s2  = diag(S2);    S2m = diag(s2);
    FS1 = compute_T(s1);
    FS2 = compute_T(s2);

    % ============================================================
    % Step 1: Q1_dot, R1_dot  (same as your code, just align symbols)
    % ============================================================
    M1   = Y_dot' * U;                  
    A_Q1 = Q1' * M1 * R1 * S1m;         
    Q1_dot = Q1 * (FS1 .* (A_Q1 + A_Q1'));

    A_R1 = S1m * Q1' * M1 * R1;        
    R1_dot = R1 * (FS1 .* (A_R1 + A_R1'));

    % ============================================================
    % Step 2: D_star_dot = dot{D}_*  (your Y_star_dot)
    % ============================================================
    Y_star_dot = Y_dot * (Q1 * R1') + Y * (Q1_dot * R1') + Y * (Q1 * R1_dot');

    % ============================================================
    % Step 3: Q2_dot, S2_dot, R2_dot
    % ============================================================

    % (I - U U^T) dot{D}_*
    PU = Y_star_dot - U * (U' * Y_star_dot);

    % M = Q2^T (I-UU^T) dot{D}_* R2
    M  = Q2' * PU * R2;

    % ----- Q2_dot -----
    A_Q2  = M * S2m;                     
    Q2_dot_main = Q2 * (FS2 .* (A_Q2 + A_Q2')); 
    S2_inv = diag(1./s2);
    Q2_dot_proj = (eye(size(Q2,1)) - Q2*Q2') * (PU * R2 * S2_inv);

    Q2_dot = Q2_dot_main + Q2_dot_proj;

    % ----- S2_dot -----
    % I_p ∘ ( Q2^T PU R2 )  -> keep diagonal only
    S2_dot_vec = diag(M);

    % ----- R2_dot -----
    A_R2  = S2m * M;                   
    R2_dot_main = R2 * (FS2 .* (A_R2 + A_R2'));
    R2_dot_proj = (eye(size(R2,1)) - R2*R2') * (PU' * Q2 * S2_inv);

    R2_dot = R2_dot_main + R2_dot_proj;

    % ============================================================
    % Step 4: Sigma_dot, xi_dot
    % ============================================================
    Sigma_dot_vec = S2_dot_vec ./ sqrt(1 - s2.^2);
    Sigma_dot = diag(Sigma_dot_vec);

    xi_dot = Q2_dot * (Sigma * R2') ...
           + Q2 * (Sigma_dot * R2') ...
           + Q2 * (Sigma * R2_dot');

end
