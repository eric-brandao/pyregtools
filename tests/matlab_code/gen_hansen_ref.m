clc; clear all; close all;
rng('default')
rng(0); % seed
%% Generate a test problem
[A,b,x_true] = gravity(64,1,0,1,0.5); b = b';% matrix, noiseless measurement, true solution
x = linspace(0, 1, length(b));
% add noise with snr
snr = 20;
n = randn(1,length(b)); n = n/norm(n); n = n.*(10^(-snr/10)).*norm(b);
b_noisy = b + n;

figure()
plot(x, b, '--k', 'LineWidth', 2); hold on;
plot(x, b_noisy); grid on;
xlim([0,1]); ylim([0, 2.5]);
legend('noiseless meas', 'noisy meas')

%% SVD
[U,s,V] = csvd(A);

%% Regularization parameters
[x_delta,lam_dp] = discrep(U,s,V,b_noisy', norm(n));
[lam_gcv,Gfun,rega_gcv] = gcv(U,s,b_noisy');
[lam_lc, rho ,eta,rega_lc] = l_curve(U,s,b_noisy');
[lam_ncp,dist,rega_ncp] = ncp(U,s,b_noisy');

%% solution
labda = 0.05;
num_svd_comp = 4;
[x_tik, rho,eta] = tikhonov(U,s,V,b_noisy', labda);
[x_k,rho,eta] = tsvd(U,s,V,b_noisy', num_svd_comp);

figure()
plot(x, x_true, '--k', 'LineWidth', 2); hold on;
plot(x, x_tik); grid on;
plot(x, x_k); grid on;
xlim([0,1]); ylim([0, 2.5]);
legend('True', 'Tikhonov', 'TSVD')

%% Save reference data
save('hansen_ref_data_gravity.mat', 'x','A', 'b', 'x_true', 'snr', 'n',...
    'b_noisy', 'lam_dp', 'lam_gcv', 'lam_lc','lam_ncp', 'x_tik', 'x_k',...
    'labda','num_svd_comp');

