%% =======================================================================
%  APPROACH 2: B-Spline Force Curves + lsqnonlin
%  =======================================================================
%  If the Kienzle model is too restrictive (e.g., forces have non-monotonic
%  behavior due to ploughing effects at small h), use B-spline basis
%  functions with ~6-8 control points per curve (12-16 total variables).
%
%  This is a good middle ground between 40 free points and 6 parametric.
% ========================================================================
clear; close all; clc;

% --- Load Experimental Data ---
%load Deney1forces.txt
%AA = Deney1forces;
dtool15=load('hardstavaxtool15datalong.txt');
%AA = dtool15(20121:20948,1:2);
%AA = dtool15(14766:15591,3:4);
%AA = dtool15(14582:15375,5:6);
%AA = dtool15(10778:11611,7:8);
%AA = dtool15(13727:14558,9:10);
%AA = dtool15(15705:16534,11:12);
%AA = dtool15(28222:29054,13:14);
%AA = dtool15(26201:27024,15:16);
AA = dtool15(15596:16423,1:18);

steps = 828;
experimental_data = AA(1:steps, :);

% --- B-Spline Configuration ---
% Use 8 control points per force curve
n_ctrl = 16;
nvars = 2 * n_ctrl;  % 16 total variables

% Control point h-locations (fixed, spanning the chip thickness range)
h_knots = linspace(0, 0.005, n_ctrl);

% Initial guess: linearly increasing forces
Ft_ctrl0 = linspace(0, 36, n_ctrl);
Fn_ctrl0 = linspace(0, 22, n_ctrl);
x0 = [Ft_ctrl0, Fn_ctrl0];

% Bounds
lb = zeros(1, nvars);
ub = 200 * ones(1, nvars);
lb(1) = 0; lb(n_ctrl+1) = 0;  % Force at h=0 should be ~0

% --- Run lsqnonlin ---
options = optimoptions('lsqnonlin', ...
    'Display', 'iter', ...
    'MaxIterations', 500, ...
    'MaxFunctionEvaluations', 10000, ...
    'StepTolerance', 1e-10, ...
    'Algorithm', 'trust-region-reflective');

residualHandle = @(params) computeResiduals_Spline(params, experimental_data, n_ctrl, h_knots);

disp('Starting B-Spline Optimization...');
tic;
[best_params, resnorm] = lsqnonlin(residualHandle, x0, lb, ub, options);
elapsed = toc;

fprintf('\nFinished in %.1f seconds, RMSE: %f\n', elapsed, sqrt(resnorm));

% --- Post-Processing ---
[~, Force_x_best, Force_y_best] = computeResiduals_Spline(best_params, experimental_data, n_ctrl, h_knots, true);

figure('Name', 'B-Spline Model Results');
subplot(2,1,1)
plot(experimental_data(:,1), 'k-', 'LineWidth', 2); hold on;
plot(Force_x_best, 'r--', 'LineWidth', 1.5);
title('Force in X-direction'); ylabel('Force [N]');
legend('Experimental', 'B-Spline Model'); grid on;

subplot(2,1,2)
plot(experimental_data(:,2), 'k-', 'LineWidth', 2); hold on;
plot(Force_y_best, 'r--', 'LineWidth', 1.5);
title('Force in Y-direction'); ylabel('Force [N]');
legend('Experimental', 'B-Spline Model'); grid on;

% Plot identified force curves
Ft_ctrl = best_params(1:n_ctrl);
Fn_ctrl = best_params(n_ctrl+1:end);
h_fine = linspace(0, 0.005, 200);
Ft_fine = interp1(h_knots, Ft_ctrl, h_fine, 'pchip');
Fn_fine = interp1(h_knots, Fn_ctrl, h_fine, 'pchip');

figure('Name', 'Identified Spline Force Curves');
plot(h_fine*1000, Ft_fine, 'b-', 'LineWidth', 2); hold on;
plot(h_fine*1000, Fn_fine, 'r-', 'LineWidth', 2);
plot(h_knots*1000, Ft_ctrl, 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot(h_knots*1000, Fn_ctrl, 'rs', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
xlabel('Uncut Chip Thickness [\mum]'); ylabel('Specific Force [N/mm]');
legend('F_t (spline)', 'F_n (spline)', 'F_t control pts', 'F_n control pts');
grid on; set(gca, 'FontSize', 14);
