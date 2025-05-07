% Calculates the L2 norm of the difference between two images
% 
% Last modified by Carlos Milovic in 2017.03.30
%
function [ rmse ] = compute_rmse( chi_recon, chi_true, mask)


rmse = 100 * norm( chi_recon(mask==1) - chi_true(mask==1) ) / norm(chi_true(mask==1));


end

