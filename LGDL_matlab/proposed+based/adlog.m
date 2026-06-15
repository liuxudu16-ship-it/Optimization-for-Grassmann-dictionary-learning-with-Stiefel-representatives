function dDelta = adlog(G,U,Y,Q1,S1,R1,Q2,S2,Sigma,R2)

%% Step 1 
A4.q = G * (R2 * Sigma);
A4.s = Q2' * G * R2;
A4.r = G' * (Q2 * Sigma);

%% Step 2 
TS2 = compute_T(S2);

LQ1 = TS2 .* (Q2' * A4.q);
LQ1 = LQ1 + LQ1';
W11=Q2*(LQ1*S2*R2');
S2_diag = diag(S2);
S2_diag(abs(S2_diag) < 1e-10) = 1e-10;  % 防止除零
S2inv = diag(1./S2_diag);
LQ2=A4.q*(S2inv*R2');
W12=LQ2-Q2*(Q2'*LQ2);
W1=W11+W12;
A3.q=W1-U*(U'*W1);

C_denom = 1 - diag(S2).^2;
C_denom(C_denom < 1e-10) = 1e-10;  % 防止除零和负数
C = 1./ sqrt(C_denom);
W2=Q2 * (diag(C).* A4.s * R2');
A3.s = W2-U*(U'*W2);                                   

LR1 = TS2 .* (R2' * A4.r);
LR1 = LR1 + LR1';
W2=Q2* (S2 * LR1 * R2'); 
LR2=Q2*(S2inv*A4.r');
LR3=LR2-LR2*(R2*R2');
A3.r=(W2+LR3)-U*(U'*(W2+LR3));

A3sum = A3.q + A3.s + A3.r;

%% Step 3
A2.y = A3sum * (R1 * Q1');
A2.q = Y' * A3sum * R1;
A2.r = A3sum' * Y * Q1;

%% Step 4
TS1 = compute_T(S1);

LQ2 = TS1 .* (Q1' * A2.q);
LQ2 = LQ2 + LQ2';
A1.q = U * (R1 * S1 * LQ2 * Q1');

LR2 = TS1 .* (R1' * A2.r);
LR2 = LR2 + LR2';
A1.r = U * (R1 * LR2 * S1 * Q1');

%% Step 5 
dDelta = A2.y + A1.q + A1.r;

end
