function [residuals, Force_x, Force_y] = computeResiduals_Spline(params, experimental_data, n_ctrl, h_knots, return_forces)
% Computes residuals using pchip-interpolated force curves from control points.
%
% INPUTS:
%   params: [Ft_ctrl(1:n_ctrl), Fn_ctrl(1:n_ctrl)] - control point force values
%   experimental_data: Nx2 experimental forces
%   n_ctrl: number of control points per curve
%   h_knots: h-values at control points (1 x n_ctrl)
%
% OUTPUTS:
%   residuals: residual vector for lsqnonlin

if nargin < 5
    return_forces = false;
end

% Unpack control points
Ft_ctrl = params(1:n_ctrl);
Fn_ctrl = params(n_ctrl+1:2*n_ctrl);

% --- Milling Simulation ---
b = 0.050;
steps = 828;
p1 = steps/2;
p2 = steps - p1;
phis1 = linspace(0, pi, p1);
phis2 = linspace(0, pi, p2);
hmax = 0.005;
rnt = 0;
hc1 = (hmax * (1 + rnt)) * sin(phis1 + sin(phis1)/100);
hc2 = (hmax * (1 - rnt)) * sin(phis2 + sin(phis2)/100);
hc = [hc1, hc2];

db = b;
Force_x = zeros(1, steps);
Force_y = zeros(1, steps);

% Pre-interpolate all chip thicknesses at once (vectorized, fast!)
% Use pchip for smooth, monotonicity-preserving interpolation
hc_positive = hc(hc > 0);
Ft_all = interp1(h_knots, Ft_ctrl, hc, 'pchip', 0);  % 0 outside range
Fn_all = interp1(h_knots, Fn_ctrl, hc, 'pchip', 0);

for i = 1:steps
    if hc(i) > 0
        Ft = Ft_all(i) * db;
        Fn = Fn_all(i) * db;

        if i <= p1
            phi_current = phis1(i);
        else
            phi_current = phis2(i - p1);
        end

        Force_x(i) = Fn*sin(phi_current) + Ft*cos(phi_current);
        Force_y(i) = -Fn*cos(phi_current) + Ft*sin(phi_current);
    end
end

% --- Residuals ---
steps = min(steps, size(experimental_data, 1));
r_x = Force_x(1:steps) - experimental_data(1:steps, 1)';
r_y = Force_y(1:steps) - experimental_data(1:steps, 2)';
residuals = [r_x, r_y]';

end
