%% Tooth-Level KDE Analysis: Fx at Fy Falling Zero
% Visualizes the probability distribution of drag forces for each tooth.
% Parameters: N=24000 rpm, fs=333kHz

clear; clc; close all;

%% 1. LOAD DATA
dtool15=load('hardstavaxtool15datalong.txt');
d1 = dtool15(1:48442,1:2);
d2 = dtool15(1:36394,3:4);
d3 = dtool15(1:34502,5:6);
d4 = dtool15(1:22845,7:8);
d5 = dtool15(1:21688,9:10);
d6 = dtool15(1:21218,11:12);
d7 = dtool15(1:33939,13:14);
d8 = dtool15(1:33008,15:16);
d9 = dtool15(1:18995,17:18);

data_cell = {d1, d2, d3, d4, d5, d6, d7,d8,d9};
titles = {'Slot 2', 'Slot 11', 'Slot 15','Slot 17', 'Slot 21', 'Slot 26','Slot 31','Slot 40','Slot 45'};
%% 2. PARAMETERS
fs = 333000;                
rpm = 24000;                
f_rot = rpm / 60;           % 400 Hz
f_tooth = f_rot * 2;        % 800 Hz
pts_per_tooth = fs / f_tooth; 
min_dist_pts = pts_per_tooth * 0.6; % Debounce
filter_cutoff = 2000;       % Filter for TRIGGER only

%% 3. ANALYSIS LOOP
for d = 1:9
    Fx = data_cell{d}(:,1);
    Fy = data_cell{d}(:,2);
    
    % --- STEP A: TRIGGER DETECTION ---
    % Filter Fy to find structural zero crossings (Pos -> Neg)
    Fy_trigger = lowpass(Fy, filter_cutoff, fs);
    crossings_idx = find(Fy_trigger(1:end-1) >= 0 & Fy_trigger(2:end) < 0);
    
    if isempty(crossings_idx)
        fprintf('Dataset %d: NO ZERO CROSSINGS. Tool likely clogged.\n', d);
        continue;
    end
    
    % Debounce
    valid_triggers = crossings_idx(1);
    for k = 2:length(crossings_idx)
        if crossings_idx(k) - valid_triggers(end) > min_dist_pts
            valid_triggers = [valid_triggers; crossings_idx(k)];
        end
    end
    
    % --- STEP B: SEPARATE TEETH ---
    Fx_sampled = Fx(valid_triggers);
    
    % Ensure even length
    if mod(length(Fx_sampled), 2) ~= 0
        Fx_sampled = Fx_sampled(1:end-1);
    end
    
    Tooth_A = Fx_sampled(1:2:end);
    Tooth_B = Fx_sampled(2:2:end);
    
    % --- STEP C: CALCULATE KDE ---
    % Kernel Density Estimation for smooth probability curves
    % We add a check to ensure we have enough points
    if length(Tooth_A) < 5
        warning('Not enough data points in Dataset %d for KDE.', d);
        continue;
    end
    
    [pdf_A, x_A] = ksdensity(Tooth_A);
    [pdf_B, x_B] = ksdensity(Tooth_B);
    
    % Calculate Stats for Annotation
    mu_A = mean(Tooth_A); std_A = std(Tooth_A);
    mu_B = mean(Tooth_B); std_B = std(Tooth_B);
    
    % --- STEP D: PLOTTING ---
    figure(d); clf;
    set(gcf, 'Position', [100, 100, 800, 500]);
    hold on;
    
    % Plot Tooth A (Blue Area)
    fill(x_A, pdf_A, 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'b', 'LineWidth', 2);
    
    % Plot Tooth B (Red Area)
    fill(x_B, pdf_B, 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'r', 'LineWidth', 2);
    
    % Add Mean Lines
    xline(mu_A, 'b--', 'LineWidth', 1.5);
    xline(mu_B, 'r--', 'LineWidth', 1.5);
    
    title([titles{d} ' - Probability Density of Fx Drag Force']);
    xlabel('Fx Magnitude (N)');
    ylabel('Probability Density');
    legend('Tooth A Distribution', 'Tooth B Distribution', 'Mean A', 'Mean B');
    grid on;
    
    % Add Run-out / Variance Annotation
    runout = abs(mu_A - mu_B);
    str = { ...
        sprintf('Run-out (Delta Mean): %.2f N', runout), ...
        sprintf('Tooth A StdDev: %.2f', std_A), ...
        sprintf('Tooth B StdDev: %.2f', std_B) ...
    };
    annotation('textbox', [0.15, 0.75, 0.3, 0.15], 'String', str, ...
        'FitBoxToText', 'on', 'BackgroundColor', 'w', 'FontSize', 10);
    
    % Auto-scale X-axis for clarity
    all_x = [x_A, x_B];
    xlim([min(all_x)-0.5, max(all_x)+0.5]);
end