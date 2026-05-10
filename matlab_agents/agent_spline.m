function [ft_ctrl, fn_ctrl, ft_max, fn_max, edge_radius_um, rmse] = agent_spline(Fx, Fy)
% B-spline force curve fitting agent.
% Ports the optimization core from run_Spline_Optimizer.m +
% computeResiduals_Spline.m. Runs with reduced iterations for speed.
%
% Inputs:
%   Fx, Fy  – force column vectors (N); uses first min(828, N) points
% Outputs:
%   ft_ctrl        – tangential force control points (1 x 16)
%   fn_ctrl        – normal force control points (1 x 16)
%   ft_max         – max tangential force from fitted curve
%   fn_max         – max normal force from fitted curve
%   edge_radius_um – tool edge radius derived from plateau-to-spike transition:
%                    h* = r_e/4  =>  r_e = 4*h*  [um]
%   rmse           – fit residual RMS error

    Fx = double(Fx(:));
    Fy = double(Fy(:));

    steps = min(828, length(Fx));
    experimental_data = [Fx(1:steps), Fy(1:steps)];

    n_ctrl   = 16;
    nvars    = 2 * n_ctrl;
    h_knots  = linspace(0, 0.005, n_ctrl);

    Ft_ctrl0 = linspace(0, 36, n_ctrl);
    Fn_ctrl0 = linspace(0, 22, n_ctrl);
    x0       = [Ft_ctrl0, Fn_ctrl0];

    lb = zeros(1, nvars);
    ub = 200 * ones(1, nvars);

    options = optimoptions('lsqnonlin', ...
        'Display',                'off', ...
        'MaxIterations',          100,   ...
        'MaxFunctionEvaluations', 2000,  ...
        'StepTolerance',          1e-6,  ...
        'Algorithm',              'trust-region-reflective');

    residualFn = @(p) computeResiduals_Spline(p, experimental_data, n_ctrl, h_knots);

    [best_params, resnorm] = lsqnonlin(residualFn, x0, lb, ub, options);

    ft_ctrl = best_params(1:n_ctrl);
    fn_ctrl = best_params(n_ctrl+1:end);
    ft_max  = max(ft_ctrl);
    fn_max  = max(fn_ctrl);
    rmse    = sqrt(resnorm / (2 * steps));

    % Find h* where Ft transitions from plateau to rapid increase.
    % Physical basis: h* = r_e/4  =>  edge_radius = 4 * h*
    h_fine   = linspace(0, 0.005, 500);
    ft_fine  = interp1(h_knots, ft_ctrl, h_fine, 'pchip');
    dft      = gradient(ft_fine, h_fine(2) - h_fine(1));

    % Skip first 10% to avoid h=0 boundary effects
    skip = max(2, round(length(h_fine) * 0.10));
    dft_work = dft(skip : end - skip);

    % Plateau = region of minimum slope
    [~, local_idx] = min(dft_work);
    plateau_idx    = local_idx + skip - 1;

    % After plateau, find where slope rises 30% of the way to its post-plateau max
    post_dft   = dft(plateau_idx : end);
    max_slope  = max(post_dft);
    threshold  = dft(plateau_idx) + (max_slope - dft(plateau_idx)) * 0.30;
    candidates = find(post_dft > threshold);

    if ~isempty(candidates)
        h_star = h_fine(plateau_idx + candidates(1) - 1);
    else
        h_star = h_fine(plateau_idx);
    end

    edge_radius_um = 4.0 * h_star * 1000.0;   % mm -> um
end
