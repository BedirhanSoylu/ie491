function wornArea()
    % Ana dizin
    baseDir = 'C:\Users\Bedirhan\Desktop\Main\Courses\Senior\IE490-491\IE491\TestData';
    
    % 1. Sıfır (Fresh) Takım Görüntüsünün Yüklenmesi
    freshImgPath = fullfile(baseDir, 'Fresh_Unworn', 'tltest0102032026_110457 AM.jpg'); % KENDI DOSYA ADINIZI YAZIN
    if exist(freshImgPath, 'file') ~= 2
        warning('Fresh tool image not found at %s. Please check the path.', freshImgPath);
    end
    img_new = imread(freshImgPath);
    if size(img_new, 3) == 3, img_new = rgb2gray(img_new); end
    
    % Yeni takım için maske oluşturma (Referans Maske)
    img_new_blur = imgaussfilt(img_new, 2);
    level_new = graythresh(img_new_blur);
    mask_new = imbinarize(img_new_blur, level_new);
    se = strel('disk', 5);
    mask_new = imclose(mask_new, se);
    mask_new = imopen(mask_new, se);
    mask_new = bwareafilt(mask_new, 1);
    
    % Hizalama (Registration) ayarları
    [optimizer, metric] = imregconfig('monomodal');
    
    % Görsel doğrulama resimlerinin otomatik kaydedileceği klasör
    outputDir = fullfile(baseDir, 'Auto_Validation_Images');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir); 
    end
    
    folderList = dir(baseDir);

    % Veri seti için dizileri kesin veri tipleriyle başlatıyoruz
    FileNames = {};                 % Metinler (String/Char) için Cell Array
    Channels = zeros(0, 1);         % Sayılar (Double) için sayısal dizi
    EdgeRadiuses = zeros(0, 1);     % Sayılar (Double) için sayısal dizi (Ortalama Yarıçap)
    ToolLengths = zeros(0, 1);      % Sayılar (Double) için sayısal dizi
    WornAreas = zeros(0, 1);        % Sayılar (Double) için sayısal dizi
    
    fprintf('Otomatik özellik çıkarımı ve görselleştirme başlatılıyor...\n');
    
    for i = 1:length(folderList)
        folderName = folderList(i).name;
        
        % Sadece "Kanal" ile başlayan klasörleri işleme al
        if folderList(i).isdir && startsWith(folderName, 'Kanal')
            channelNum = str2double(erase(folderName, 'Kanal'));
            imageFiles = dir(fullfile(baseDir, folderName, '*.jpg')); 
            
            for j = 1:length(imageFiles)
                imgName = imageFiles(j).name;
                imgPath = fullfile(baseDir, folderName, imgName);
                
                img_worn = imread(imgPath);
                if size(img_worn, 3) == 3, img_worn = rgb2gray(img_worn); end
                
                % Ortak İndeks Numarası
                idx = length(FileNames) + 1;
                
                % --- 2. Maske Çıkarma ---
                img_worn_blur = imgaussfilt(img_worn, 2);
                level_worn = graythresh(img_worn_blur);
                mask_worn = imbinarize(img_worn_blur, level_worn);
                
                mask_worn = imclose(mask_worn, se);
                mask_worn = imopen(mask_worn, se);
                mask_worn = bwareafilt(mask_worn, 1);
                
                % --- 3. Takım Uzunluğu (Tool Length) ---
                [rows, cols] = find(mask_worn);
                if ~isempty(cols)
                    [~, minColIdx] = min(cols);
                    [~, maxColIdx] = max(cols);
                    leftmost = [cols(minColIdx), rows(minColIdx)];
                    rightmost = [cols(maxColIdx), rows(maxColIdx)];
                    tool_length = norm(leftmost - rightmost);
                else
                    tool_length = NaN; 
                end
                
                % --- 4. Aşınmış Alan (Worn Area) ---
                try
                    tform = imregtform(uint8(mask_worn), uint8(mask_new), 'rigid', optimizer, metric);
                    aligned_mask_worn = imwarp(uint8(mask_worn), tform, 'OutputView', imref2d(size(mask_new)));
                    aligned_mask_worn = logical(aligned_mask_worn);
                    
                    worn_area_mask = mask_new & ~aligned_mask_worn;
                    worn_pixels_count = sum(worn_area_mask(:));
                catch
                    worn_pixels_count = NaN; 
                end
                
                % --- 5. Edge Radius Hesaplama (Entegre RANSAC Mantığı) ---
                try
                    [R_L, xc_L, yc_L, R_R, xc_R, yc_R] = calculate_edge_radii(imgPath);
                    % İki tarafın ortalamasını alarak datasete kaydediyoruz
                    current_edge_radius = mean([R_L, R_R], 'omitnan');
                    if isempty(current_edge_radius) || isnan(current_edge_radius)
                        current_edge_radius = NaN;
                    end
                catch
                    current_edge_radius = NaN;
                    R_L = NaN; R_R = NaN;
                end
                
                % --- 6. GÜVENLİ VERİ KAYDEDİCİ ---
                FileNames{idx, 1} = imgName;
                Channels(idx, 1) = channelNum;
                EdgeRadiuses(idx, 1) = current_edge_radius; 
                ToolLengths(idx, 1) = tool_length(1);
                WornAreas(idx, 1) = worn_pixels_count(1);
                
                % --- 7. Görselleştirme ---
                fig = figure('Visible', 'off'); 
                imshow(img_worn); hold on;
                
                % 1. SIFIR TAKIM MASKESİ
                if exist('tform', 'var')
                    tform_inv = invert(tform);
                    mask_new_projected = imwarp(mask_new, tform_inv, 'OutputView', imref2d(size(img_worn)));
                    red_overlay = cat(3, ones(size(img_worn)), zeros(size(img_worn)), zeros(size(img_worn)));
                    h_img = imshow(red_overlay);
                    set(h_img, 'AlphaData', mask_new_projected * 0.4); 
                end

                % Aşınmış maske sınırları
                visboundaries(mask_worn, 'Color', 'y', 'LineWidth', 1);
                
                % 2. TAKIM UZUNLUĞU 
                if ~isnan(tool_length) && tool_length > 0
                    plot([leftmost(1), rightmost(1)], [leftmost(2), rightmost(2)], 'r-', 'LineWidth', 2.5);
                    plot(leftmost(1), leftmost(2), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
                    plot(rightmost(1), rightmost(2), 'bs', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
                    
                    mid_x = (leftmost(1) + rightmost(1)) / 2;
                    mid_y = (leftmost(2) + rightmost(2)) / 2;
                    text(mid_x, mid_y - 25, sprintf('Length: %.2f px', tool_length),...
                        'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold',...
                        'BackgroundColor', 'white', 'EdgeColor', 'black',...
                        'HorizontalAlignment', 'center');
                end
                
                % 3. KÖŞE YARIÇAPI ÇEMBERLERİ (Sol ve Sağ)
                if ~isnan(R_L) && R_L > 0
                    viscircles([xc_L, yc_L], R_L, 'Color', 'g', 'LineWidth', 1.5);
                    plot(xc_L, yc_L, 'g+', 'MarkerSize', 10, 'LineWidth', 2);
                end
                if ~isnan(R_R) && R_R > 0
                    viscircles([xc_R, yc_R], R_R, 'Color', 'g', 'LineWidth', 1.5);
                    plot(xc_R, yc_R, 'g+', 'MarkerSize', 10, 'LineWidth', 2);
                end
                
                title(sprintf('Kanal: %d | Worn Area: %d px | Avg R: %.1f px', channelNum, worn_pixels_count, current_edge_radius), 'Interpreter', 'none');
                hold off;
                
                saveName = fullfile(outputDir, sprintf('Kanal%d_%s', channelNum, imgName));
                saveas(fig, saveName);
                close(fig);
            end
            fprintf('Kanal %d başarıyla işlendi.\n', channelNum);
        end
    end
    
    % --- 8. Tablo Oluşturma ve Excel ---
    datasetTable = table(FileNames, Channels, EdgeRadiuses, ToolLengths, WornAreas,...
        'VariableNames', {'FileName', 'Channel', 'EdgeRadius', 'ToolLength', 'WornArea'});
        
    excelFileName = fullfile(baseDir, 'Tool_Features_Dataset.xlsx');
    writetable(datasetTable, excelFileName);
    
    fprintf('\nOtomasyon tamamlandı!\n- Çıktılar (Excel): %s\n- Gözle doğrulama görselleri: %s klasörüne kaydedildi.\n', excelFileName, outputDir);
end

% =========================================================
% ENTEGRE RANSAC EDGE RADIUS FONKSIYONLARI
% =========================================================

function [R_L, xc_L, yc_L, R_R, xc_R, yc_R] = calculate_edge_radii(imgPath)
    % Initialize default NaN outputs in case of failure
    R_L = NaN; xc_L = NaN; yc_L = NaN;
    R_R = NaN; xc_R = NaN; yc_R = NaN;
    
    config.ANGLE_REF = 100;
    
    img_raw = imread(imgPath);
    if size(img_raw, 3) == 3
        img_gray = rgb2gray(img_raw);
    else
        img_gray = img_raw;
    end
    
    [h, w] = size(img_gray);
    mean_lum = mean(img_gray(:)); 
    config.BLUR_COEFF = 0.9 + (-mean_lum/255 + 1) * 0.1; 
    
    blur = imgaussfilt(img_gray, 2); 
    level = graythresh(blur) * config.BLUR_COEFF;
    bw = imbinarize(blur, level);
    
    se_size = max(3, round(w * 0.01)); 
    se = strel('disk', se_size);
    bw = imclose(bw, se);
    bw = imopen(bw, se);

    cc = bwconncomp(bw);
    stats = regionprops(cc, 'Area', 'BoundingBox', 'PixelIdxList');
    mask = false(size(bw));
    tool_center_x = floor(w / 2); 
    tool_w = 0;

    if ~isempty(stats)
        areas = [stats.Area];
        min_area = max(500, 0.01 * h * w);
        [max_area, largest_idx] = max(areas);
        
        if max_area > min_area 
            mask(stats(largest_idx).PixelIdxList) = true;
            bb = stats(largest_idx).BoundingBox; 
            tool_w = bb(3); 
            tool_center_x = round(bb(1) + bb(3)/2);
        end
    end    

    if tool_w == 0
        return; % Mask failed
    end

    config.RANSAC_ITERATIONS = 40;  
    config.RANSAC_TOLERANCE = max(1.2, tool_w * 0.006);  
    config.RANSAC_CENTER_DELTA = floor(tool_w * 0.02); 
    config.RANSAC_CENTER_GRID = 5; 
    config.ROI_MARGIN = floor(tool_w * 0.05);
    
    mask = imfill(mask, 'holes');
    masked_gray = img_gray;
    masked_gray(~mask) = 0; 
    tool_center_x = max(1, min(tool_center_x, w));
    
    img_left_raw = masked_gray(:, 1:tool_center_x);
    img_right_raw = masked_gray(:, (tool_center_x + 1):end);
    
    % --- LEFT PART ---
    left_canny = edge(img_left_raw, 'Canny', [0.2, 0.65]);
    mask_left = mask(:, 1:tool_center_x);
    try
        [R_L, xc_L, yc_L] = analyze_single_side_silent(left_canny, mask_left, config, tool_w);
    catch
        R_L = NaN; xc_L = NaN; yc_L = NaN;
    end

    % --- RIGHT PART ---
    img_right_flipped = flip(img_right_raw, 2);
    img_right_flipped = flip(img_right_flipped, 1);
    mask_right = mask(:, (tool_center_x + 1):end);
    mask_right_flipped = flip(flip(mask_right, 2), 1);
    right_canny = edge(img_right_flipped, 'Canny', [0.2, 0.65]);
    
    try
        [R_R_flip, xc_R_flip, yc_R_flip] = analyze_single_side_silent(right_canny, mask_right_flipped, config, tool_w);
        
        % Un-flip coordinates to match original image space!
        if ~isnan(R_R_flip)
            w_right = size(img_right_raw, 2);
            yc_R = h - yc_R_flip + 1;
            xc_R = tool_center_x + (w_right - xc_R_flip + 1);
            R_R = R_R_flip;
        end
    catch
        R_R = NaN; xc_R = NaN; yc_R = NaN;
    end
end

function [R_geo, xc_geo, yc_geo] = analyze_single_side_silent(img_analysis, side_mask, cfg, tw)
    R_geo = NaN; xc_geo = NaN; yc_geo = NaN;
    ANGLE_REF = cfg.ANGLE_REF; 
    
    [all_y, all_x] = find(img_analysis > 0); 
    if isempty(all_x), return; end

    anchorPt = findRefinedAnchor(all_x, all_y, ANGLE_REF, tw);
    y_center = mean(all_y);
    top_half_indices = find(all_y < y_center);
    
    if isempty(top_half_indices)
        top_half_indices = 1:length(all_y);
    end
    
    top_half_y = all_y(top_half_indices);
    top_half_x = all_x(top_half_indices);
    min_y_val = min(top_half_y);
    on_top_line_indices = find(top_half_y <= (min_y_val + tw * 0.03));
    
    if isempty(on_top_line_indices)
         [~, min_idx] = min(top_half_y);
         topLeftPt = [top_half_x(min_idx), top_half_y(min_idx)];
    else
        [~, leftest_idx] = min(top_half_x(on_top_line_indices));
        actual_idx = on_top_line_indices(leftest_idx);
        topLeftPt = [top_half_x(actual_idx), top_half_y(actual_idx)];
    end
    
    if topLeftPt(2) > anchorPt(2)
        topLeftPt = anchorPt;
    end
    
    significant_dist_threshold = tw * 0.07; 
    current_dist = sqrt(sum((topLeftPt - anchorPt).^2));
    if current_dist > significant_dist_threshold
        topLeftPt = anchorPt;
    end

    ROI_MARGIN = cfg.ROI_MARGIN; 
    x_limit_min = min(anchorPt(1), topLeftPt(1)) - ROI_MARGIN;
    x_limit_max = max(anchorPt(1), topLeftPt(1)) + ROI_MARGIN;
    y_limit_min = min(anchorPt(2), topLeftPt(2)) - ROI_MARGIN;
    y_limit_max = max(anchorPt(2), topLeftPt(2)) + ROI_MARGIN;
    
    roi_mask = (all_x >= x_limit_min) & (all_x <= x_limit_max) & ...
               (all_y >= y_limit_min) & (all_y <= y_limit_max);
    xs_full = all_x(roi_mask);
    ys_full = all_y(roi_mask);
    
    if length(xs_full) < 3, return; end

    points = [xs_full(:), ys_full(:)];
    numPoints = size(points, 1);
    isDominated = false(numPoints, 1); 
    for i = 1:numPoints
        for j = 1:numPoints
            if i == j; continue; end
            x_better = (points(j, 1) <= points(i, 1)); 
            y_better = (points(j, 2) <= points(i, 2));
            x_strict = (points(j, 1) < points(i, 1));
            y_strict = (points(j, 2) < points(i, 2));
            if x_better && y_better && (x_strict || y_strict)
                isDominated(i) = true; break;
            end
        end
    end
    xs = points(~isDominated, 1); ys = points(~isDominated, 2);
    if length(xs) < 2, return; end

    xc_initial = topLeftPt(1); 
    yc_initial = anchorPt(2);
    
    [xs_fit, ys_fit, xc_geo, yc_geo] = ransac_filter_interval(xs, ys, xc_initial, yc_initial, side_mask, cfg);

    if length(xs_fit) < 2
        xs_fit = xs;
        ys_fit = ys;
    end
    
    % Final Circular Radius Calculation
    R_geo = circfit_radius_only(xs_fit, ys_fit, xc_geo, yc_geo);
end

function P_new = findRefinedAnchor(all_x, all_y, angle_deg, tw)
    left_offset = tw * 0.01;
    leftmost_indices = find(all_x <= min(all_x) + left_offset);
    [~, top_sub_idx] = min(all_y(leftmost_indices));
    actual_start_idx = leftmost_indices(top_sub_idx);
    
    x_start = all_x(actual_start_idx); 
    y_start = all_y(actual_start_idx);
    m = -tand(angle_deg);
    A = m; B = -1; C = y_start - m * x_start;
    norm_val = sqrt(A^2 + B^2);
    dist = abs(A * all_x + B * all_y + C) / norm_val;
    band_indices = find(dist <= 3); 
    if isempty(band_indices), P_new = [x_start, y_start]; return; end
    x_band = all_x(band_indices); y_band = all_y(band_indices);
    [~, min_y_idx] = min(y_band);
    P_new = [x_band(min_y_idx), y_band(min_y_idx)];
end

function [xs_filtered, ys_filtered, best_xc, best_yc] = ransac_filter_interval(xs, ys, xc_initial, yc_initial, side_mask, cfg)
    num_points = length(xs);
    [mask_h, mask_w] = size(side_mask);
    if num_points < 2
        xs_filtered = []; ys_filtered = [];
        best_xc = xc_initial; best_yc = yc_initial;
        return;
    end
    
    delta = cfg.RANSAC_CENTER_DELTA;
    grid_size = cfg.RANSAC_CENTER_GRID;
    center_range = linspace(0, delta, grid_size); 
    
    [dx, dy] = meshgrid(center_range, center_range);
    candidate_xc = xc_initial + dx(:);
    candidate_yc = yc_initial + dy(:);
    num_candidates = length(candidate_xc);
    
    best_inlier_count = 0;
    best_inliers_mask = false(num_points, 1);
    best_xc = xc_initial;
    best_yc = yc_initial;
    
    for c = 1:num_candidates
        xc_c = candidate_xc(c);
        yc_c = candidate_yc(c);
        ix = round(xc_c); iy = round(yc_c);
        if ix >= 1 && ix <= mask_w && iy >= 1 && iy <= mask_h
            if side_mask(iy, ix) == 0 
                continue;
            end
        else
            continue; 
        end
        [current_inliers_mask, current_inlier_count] = run_ransac_fixed_center(xs, ys, xc_c, yc_c, cfg.RANSAC_TOLERANCE, cfg.RANSAC_ITERATIONS);
        
        if current_inlier_count > best_inlier_count
            best_inlier_count = current_inlier_count;
            best_inliers_mask = current_inliers_mask;
            best_xc = xc_c;
            best_yc = yc_c;
        end
    end
    
    if best_inlier_count > 1
        xs_filtered = xs(best_inliers_mask);
        ys_filtered = ys(best_inliers_mask);
    else
        xs_filtered = [];
        ys_filtered = [];
    end
end

function [inliers_mask, inlier_count] = run_ransac_fixed_center(xs, ys, xc_fixed, yc_fixed, tolerance, max_iterations)
    num_points = length(xs);
    best_inlier_count = 0;
    inliers_mask = false(num_points, 1);
    
    for iter = 1:max_iterations
        sample_idx = randi(num_points, 1);
        x_sample = xs(sample_idx);
        y_sample = ys(sample_idx);
        
        R_hyp = sqrt((x_sample - xc_fixed)^2 + (y_sample - yc_fixed)^2);
        distances_to_center = sqrt((xs - xc_fixed).^2 + (ys - yc_fixed).^2);
        
        current_inliers_mask = abs(distances_to_center - R_hyp) <= tolerance;
        current_inlier_count = sum(current_inliers_mask);
        
        if current_inlier_count > best_inlier_count
            best_inlier_count = current_inlier_count;
            inliers_mask = current_inliers_mask;
        end
    end
    inlier_count = best_inlier_count;
end

function R = circfit_radius_only(x, y, xc_geo, yc_geo)
    x = x(:); y = y(:);
    squared_distances = (x - xc_geo).^2 + (y - yc_geo).^2;
    R = sqrt(sum(squared_distances) / length(x));
end