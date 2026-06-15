function P = ProjGr(X, Z)
% Projection onto tangent space to Grassmann manifold
P = Z - X*(X'*Z);