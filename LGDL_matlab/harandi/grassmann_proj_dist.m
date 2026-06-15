function dist = grassmann_proj_dist(SY1, SY2)
% grassmann_proj_dist  Compute projection distance between Grassmann points
%
% Usage:
%   D = grassmann_proj_dist(X)        % self distance matrix
%   D = grassmann_proj_dist(X, Y)     % cross distance matrix
%
% Input:
%   SY1 : d×p×n1 array of orthonormal bases
%   SY2 : d×p×n2 array of orthonormal bases (optional)
%
% Output:
%   dist : n2×n1 distance matrix
%
% Note: Projection distance d(X,Y) = p - ||X'Y||_F^2
%       where p is the subspace dimension

p = size(SY1, 2);

if nargin < 2
    % Self distance
    kernel = grassmann_proj_local(SY1);
else
    % Cross distance
    kernel = grassmann_proj_local(SY1, SY2);
end

dist = p - kernel;

end

%% Local projection kernel function
function dist_p = grassmann_proj_local(SY1, SY2)

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




