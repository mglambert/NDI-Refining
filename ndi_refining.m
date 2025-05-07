clearvars
close all
clc
%% Load Data
addpath('./Data/');
addpath('./matlab_files/');
addpath('./matlab_files/nii/');

load('mask_final.mat');
load('QSMnetp_recon.mat');
load('xQSM_recon.mat');
mask = mask_final;
clear mask_final;

phs_scale =  42.7747892 * 7;
N = [256, 256, 256];

freqmap = load_nii('./Data/Frequency.nii.gz').img;
gtsim2 = load_nii('./Data/Sim2ChiGT.nii.gz').img;

%% QSMnetp

rmse_qsmnet1 = compute_rmse(QSMnetp_recon, gtsim2, mask);
disp(['QSMnet+ Base RMSE ', num2str(rmse_qsmnet1)]);

%%
params = {};
params.input = zeros(N);
params.input(1:164, 1:205, 1:205) = freqmap / phs_scale;

params.K = dipole_kernel(size(params.input), [1,1,1], 0);

params.isPrecond = true;
params.precond = zeros(N);

params.precond(1:164, 1:205, 1:205) = QSMnetp_recon.*mask;

params.GT = zeros(N);
params.GT(1:164, 1:205, 1:205) = gtsim2 .* mask;

params.mask = zeros(N);
params.mask(1:164, 1:205, 1:205) = mask;

params.maxOuterIter = 5;

params.tau = 2;

outndi = ndi(params);

imagesc3d22(outndi.x(1:164, 1:205, 1:205), ['QSMnet+ ', num2str(rmse_qsmnet1) , ' to ', num2str(outndi.rmse(end))], [90, 90, 90], [-0.1, 0.1]);

%% xQSM 

rmse_xqsm1 = compute_rmse(xQSM_recon, gtsim2, mask);
disp(['xQSM Base RMSE ', num2str(rmse_xqsm1)]);

%%
params = {};
params.input = zeros(N);
params.input(1:164, 1:205, 1:205) = freqmap / phs_scale;

params.K = dipole_kernel(size(params.input), [1,1,1], 0);

params.isPrecond = true;
params.precond = zeros(N);

params.precond(1:164, 1:205, 1:205) = xQSM_recon.*mask;

params.GT = zeros(N);
params.GT(1:164, 1:205, 1:205) = gtsim2 .* mask;

params.mask = zeros(N);
params.mask(1:164, 1:205, 1:205) = mask;

params.maxOuterIter = 9;

params.tau = 2;

outndi = ndi(params);

imagesc3d22(outndi.x(1:164, 1:205, 1:205), ['xQSM ', num2str(rmse_xqsm1) , ' to ', num2str(outndi.rmse(end))], [90, 90, 90], [-0.1, 0.1]);


%% NDI alone

params = {};
params.input = zeros(N);
params.input(1:164, 1:205, 1:205) = freqmap / phs_scale;

params.K = dipole_kernel(size(params.input), [1,1,1], 0);

params.GT = zeros(N);
params.GT(1:164, 1:205, 1:205) = gtsim2 .* mask;

params.mask = zeros(N);
params.mask(1:164, 1:205, 1:205) = mask;

params.maxOuterIter = 12;

params.tau = 2;

outndi = ndi(params);

imagesc3d22(outndi.x(1:164, 1:205, 1:205), ['NDI solo ', num2str(outndi.rmse(end))], [90, 90, 90], [-0.1, 0.1]);
