%% Tooth-Level Analysis: Fx sampled at Fy Falling Zero
% 1. Filter Fy to remove chatter.
% 2. Find every instance where Fy drops below Absolute Zero.
% 3. Record Fx at that moment.

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
f_tooth = f_rot * 2;        % 800 Hz (Tooth frequency)

% Debounce Logic: 
% We expect a crossing every ~416 points. 
% We ignore new triggers for 60% of that duration (~250 pts) to avoid noise.
pts_per_tooth = fs / f_tooth; 
min_dist_pts = pts_per_tooth * 0.6; 

% Filter Fy for triggering ONLY. 
% 2500 Hz allows the 800Hz tooth wave to pass but kills high-freq chatter.
filter_cutoff = 2500;       

%% 3. ANALYSIS LOOP
fprintf('%-25s | %-15s | %-15s | %-15s\n', ...
    'Slot', 'Avg Fx (Tooth A)', 'Avg Fx (Tooth B)', 'Runout Diff');
fprintf('%s\n', repmat('-', 1, 80));

for d = 1:9
    Fx = data_cell{d}(:,1);
    Fy = data_cell{d}(:,2);
    
    % --- STEP A: PREPARE TRIGGER SIGNAL (Fy) ---
    % Lowpass Fy to see the "structural" zero crossings clearly
    Fy_trigger = lowpass(Fy, filter_cutoff, fs);
    
    % --- STEP B: DETECT FALLING EDGE (Pos -> Neg) ---
    % Current point >= 0, Next point < 0
    crossings_idx = find(Fy_trigger(1:end-1) >= 0 & Fy_trigger(2:end) < 0);
    
    % --- STEP C: DEBOUNCE (Isolate individual teeth) ---
    if isempty(crossings_idx)
        fprintf('%-25s | NO ZERO CROSSINGS FOUND ON Fy\n', titles{d});
        figure(d); clf;
        plot(Fy); title([titles{d} ' - No Zero Crossings!']); ylabel('Fy');
        continue;
    end
    
    valid_triggers = crossings_idx(1);
    for k = 2:length(crossings_idx)
        if crossings_idx(k) - valid_triggers(end) > min_dist_pts
            valid_triggers = [valid_triggers; crossings_idx(k)];
        end
    end
    
    % --- STEP D: SAMPLE Fx ---
    Fx_sampled = Fx(valid_triggers);
    
    % Statistics for alternating teeth
    % Assume Start = Tooth A, Next = Tooth B, etc.
    % Truncate to even number for fair comparison
    len = length(Fx_sampled);
    if mod(len, 2) ~= 0
        Fx_calc = Fx_sampled(1:end-1);
    else
        Fx_calc = Fx_sampled;
    end
    
    Tooth_A = Fx_calc(1:2:end);
    Tooth_B = Fx_calc(2:2:end);
    
    fprintf('%-25s | %-15.4f | %-15.4f | %-15.4f\n', ...
        titles{d}, mean(Tooth_A), mean(Tooth_B), abs(mean(Tooth_A)-mean(Tooth_B)));
    
    % --- PLOTTING ---
    figure(d); clf;
    set(gcf, 'Position', [100, 100, 1000, 700]);
    sgtitle(titles{d}, 'FontSize', 16, 'FontWeight', 'bold');
    
    % Subplot 1: Fy Trace (The Trigger Source)
    subplot(3,1,1);
    t_axis = (0:length(Fy)-1)/fs * 1000;
    plot(t_axis, Fy, 'Color', [1 0.7 0.7]); hold on; % Raw Fy
    plot(t_axis, Fy_trigger, 'r', 'LineWidth', 1.5); % Filtered Fy
    yline(0, 'k-');
    
    % Plot Red Dots where Fy crosses zero
    trig_t = valid_triggers / fs * 1000;
    plot(trig_t, zeros(size(trig_t)), 'ro', 'MarkerFaceColor', 'k');
    
    title('1. Trigger Source: Fy Falling Edge (Pos \rightarrow Neg)');
    ylabel('Fy (N)');
    xlim([0, 10]); % Zoom in
    grid on;
    
    % Subplot 2: Fx Trace (The Result)
    subplot(3,1,2);
    plot(t_axis, Fx, 'Color', [0.7 0.7 1]); hold on; % Raw Fx
    
    % Plot Blue/Magenta markers for Fx values
    is_tooth_A = mod(1:length(Fx_sampled), 2) == 1;
    
    plot(trig_t(is_tooth_A), Fx_sampled(is_tooth_A), ...
        'bv', 'MarkerFaceColor', 'b', 'MarkerSize', 6);
    plot(trig_t(~is_tooth_A), Fx_sampled(~is_tooth_A), ...
        'mv', 'MarkerFaceColor', 'm', 'MarkerSize', 6);
    
    title('2. Fx Value at the moment Fy crosses Zero');
    ylabel('Fx (N)');
    xlim([0, 10]);
    grid on;
    legend('Raw Fx', 'Tooth A', 'Tooth B');
    
    % Subplot 3: Bar Chart Comparison
    subplot(3,1,3);
    b = bar(1:length(Fx_sampled), Fx_sampled, 'FaceColor', 'flat');
    
    % Color Logic: Blue = Tooth A, Magenta = Tooth B
    b.CData = repmat([1 0 1], length(Fx_sampled), 1); % Default Magenta
    b.CData(1:2:end, :) = repmat([0 0 1], length(1:2:length(Fx_sampled)), 1); % Odd = Blue
    b.FaceAlpha = 0.6;
    
    title('Tooth-by-Tooth Drag Force (Fx when Fy=0)');
    xlabel('Tooth Pass Index');
    ylabel('Fx Magnitude (N)');
    grid on;
end