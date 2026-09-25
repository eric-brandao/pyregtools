clc; clear all; close all;
rng('default')
rng(0); % seed
%% Generate a test problem
[A,b,x_true] = gravity(128,1,0,1,0.5); b = b';% matrix, noiseless measurement, true solution
%[A,b,x_true] = shaw(128); b = b'; % matrix, noiseless measurement, true solution
x = linspace(0, 1, length(b));
% add noise with snr
snr = 20;
n = randn(1,length(b)); n = n/norm(n); n = n.*(10^(-snr/10)).*norm(b);
b_noisy = b + n;

%% Generate iterative solver solution
num_iterations = 20;
[X,rho,eta] = cgls(A,b_noisy',num_iterations);
error_hist = zeros(1,num_iterations);
for k = 1:num_iterations
    error_hist(k) = norm(X(:,k)-x_true)/norm(x_true);
end
%% 
figure()
plot(x, b, '--k', 'LineWidth', 2); hold on;
plot(x, b_noisy); grid on;
xlim([0,1]); ylim([0, 1.1*max(b_noisy)]);
legend('noiseless meas', 'noisy meas')

figure()
plot(error_hist, 'o-k'); grid on; ylim([0,1.5]);
xlabel("Iterations"); ylabel("Error")
title("Error history"); 

figure()
plot(rho, eta, 'o-k'); grid on;
xlabel("Residual norm"); ylabel("Solution norm")
title("L-curve behavior")

figure()
plot(x, x_true, '--k', 'LineWidth', 2); hold on;
plot(x, X(:,1)); grid on;
plot(x, X(:,4)); grid on;
xlim([0,1]); ylim([0, 2.5]);
legend('True', '1st Iteration', 'Last')


%%
save('hansen_ref_data_gravity_is.mat', 'x','A', 'b', 'x_true', 'snr', 'n',...
    'b_noisy', 'X', 'rho', 'eta','num_iterations', 'error_hist');