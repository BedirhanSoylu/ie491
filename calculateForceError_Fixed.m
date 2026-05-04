function [obj, Force_x, Force_y] = calculateForceError_Fixed(force_vars, experimental_data, h_values, return_forces)
% Fixed version: uses pchip interpolation instead of GPR for massive speedup.
%
% INPUTS:
%   force_vars: 1x40 vector [Ft(1:20), Fn(1:20)]
%   experimental_data: Nx2 matrix of experimental [Fx, Fy]
%   h_values: 20x1 vector of chip thickness values
%   return_forces: (optional) if true, also return Force_x, Force_y
%
% OUTPUTS:
%   obj: RMSE error
%   Force_x, Force_y: simulated forces (optional)

if nargin < 4
    return_forces = false;
end

% --- Unpack ---
Ft_values = force_vars(1:16);
Fn_values = force_vars(17:32);

% --- Milling Simulation ---
b = 0.050;
steps = 828;
p1 = 414;
p2 = steps - p1;
phis1 = linspace(0, pi, p1);
phis2 = linspace(0, pi, p2);
hmax = 0.005;
rnt = 0;
hc1 = (hmax * (1 + rnt)) * sin(phis1 + sin(phis1)/100);
hc2 = (hmax * (1 - rnt)) * sin(phis2 + sin(phis2)/100);
hc = [hc1, hc2];

db = b;

% FAST interpolation using pchip (replaces GPR - orders of magnitude faster)
% pchip preserves shape and is smooth (C1 continuous)
Ft_interp = interp1(h_values, Ft_values, hc, 'pchip', 0);
Fn_interp = interp1(h_values, Fn_values, hc, 'pchip', 0);

Force_x = zeros(1, steps);
Force_y = zeros(1, steps);

for i = 1:steps
    if hc(i) > 0
        Ft = Ft_interp(i) * db;
        Fn = Fn_interp(i) * db;

        if i <= p1
            phi_current = phis1(i);
        else
            phi_current = phis2(i - p1);
        end

        Force_x(i) = Fn*sin(phi_current) + Ft*cos(phi_current);
        Force_y(i) = -Fn*cos(phi_current) + Ft*sin(phi_current);
    end
end

% --- Error ---
obj1 = (Force_x - experimental_data(1:steps, 1)').^2;
obj2 = (Force_y - experimental_data(1:steps, 2)').^2;
obj = sqrt(sum(obj1) + sum(obj2));

end
