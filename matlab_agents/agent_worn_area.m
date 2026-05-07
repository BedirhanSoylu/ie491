function [edge_radius_px, tool_length_px, worn_area_px, ideal_worn_area_px] = agent_worn_area(worn_img_path, fresh_img_path)
% Tool wear feature extraction for a single image pair.
% Ports the core image processing logic from wornArea.m as a callable
% function. All helper sub-functions are included locally.
%
% Inputs:
%   worn_img_path        – full path to the worn tool image (char/string)
%   fresh_img_path       – full path to the fresh reference image (char/string)
% Outputs:
%   edge_radius_px       – mean RANSAC-fitted edge radius in pixels
%   tool_length_px       – leftmost-to-rightmost pixel distance
%   worn_area_px         – pixel count of worn region (fresh minus worn mask)
%   ideal_worn_area_px   – theoretical ideal worn area: r^2*(cos^3(40)/sin(40)+sin(40)*cos(40)-5*pi/18)

    worn_img_path  = char(worn_img_path);
    fresh_img_path = char(fresh_img_path);

    IDEAL_AREA_CONST = cosd(40)^3/sind(40) + sind(40)*cosd(40) - 5*pi/18;

    edge_radius_px       = NaN;
    tool_length_px       = NaN;
    worn_area_px         = NaN;
    ideal_worn_area_px   = NaN;

    % --- Fresh reference mask ---
    img_new = imread(fresh_img_path);
    if size(img_new, 3) == 3, img_new = rgb2gray(img_new); end
    img_new(img_new > prctile(double(img_new(:)), 95)) = uint8(prctile(double(img_new(:)), 95));
    img_new      = medfilt2(img_new, [3 3]);
    img_new      = imadjust(img_new);
    img_new_blur = imgaussfilt(img_new, 3);
    level_new    = graythresh(img_new_blur);
    mask_new     = imbinarize(img_new_blur, level_new);
    se = strel('disk', 4);
    mask_new = imclose(mask_new, se);
    mask_new = imopen(mask_new, se);
    mask_new = bwareafilt(mask_new, 1);
    mask_new = imfill(mask_new, 'holes');

    [optimizer, metric] = imregconfig('monomodal');

    % --- Worn image mask ---
    img_worn = imread(worn_img_path);
    if size(img_worn, 3) == 3, img_worn = rgb2gray(img_worn); end
    img_worn(img_worn > prctile(double(img_worn(:)), 95)) = uint8(prctile(double(img_worn(:)), 95));
    img_worn      = medfilt2(img_worn, [3 3]);
    img_worn      = imadjust(img_worn);
    img_worn_blur = imgaussfilt(img_worn, 3);
    level_worn    = graythresh(img_worn_blur);
    mask_worn     = imbinarize(img_worn_blur, level_worn);
    mask_worn = imclose(mask_worn, se);
    mask_worn = imopen(mask_worn, se);
    mask_worn = bwareafilt(mask_worn, 1);
    mask_worn = imfill(mask_worn, 'holes');

    % --- Tool Length ---
    [rows, cols] = find(mask_worn);
    if ~isempty(cols)
        [~, minColIdx] = min(cols);
        [~, maxColIdx] = max(cols);
        leftmost  = [cols(minColIdx), rows(minColIdx)];
        rightmost = [cols(maxColIdx), rows(maxColIdx)];
        tool_length_px = norm(leftmost - rightmost);
    end

    % --- Worn Area (image registration) ---
    try
        tform = imregtform(uint8(mask_worn), uint8(mask_new), 'rigid', optimizer, metric);
        aligned = imwarp(uint8(mask_worn), tform, 'OutputView', imref2d(size(mask_new)));
        worn_area_mask = mask_new & ~logical(aligned);
        worn_area_px   = sum(worn_area_mask(:));
    catch
        worn_area_px = NaN;
    end

    % --- Edge Radius (RANSAC) ---
    try
        [R_L, ~, ~, R_R, ~, ~] = calc_edge_radii(worn_img_path);
        edge_radius_px = mean([R_L, R_R], 'omitnan');
        if isempty(edge_radius_px) || isnan(edge_radius_px)
            edge_radius_px = NaN;
        end
    catch
        edge_radius_px = NaN;
    end

    % --- Ideal Worn Area ---
    if ~isnan(edge_radius_px) && edge_radius_px > 0
        ideal_worn_area_px = edge_radius_px^2 * IDEAL_AREA_CONST;
    end
end

% =============================================================
% LOCAL HELPER FUNCTIONS  (ported verbatim from wornArea.m)
% =============================================================

function [R_L, xc_L, yc_L, R_R, xc_R, yc_R] = calc_edge_radii(imgPath)
    R_L = NaN; xc_L = NaN; yc_L = NaN;
    R_R = NaN; xc_R = NaN; yc_R = NaN;

    cfg.ANGLE_REF = 100;
    img_raw = imread(imgPath);
    if size(img_raw, 3) == 3
        img_gray = rgb2gray(img_raw);
    else
        img_gray = img_raw;
    end
    [h, w]      = size(img_gray);
    mean_lum    = mean(img_gray(:));
    cfg.BLUR_COEFF = 0.9 + (-mean_lum/255 + 1) * 0.1;

    blur  = imgaussfilt(img_gray, 2);
    level = graythresh(blur) * cfg.BLUR_COEFF;
    bw    = imbinarize(blur, level);
    se_sz = max(3, round(w * 0.01));
    se    = strel('disk', se_sz);
    bw    = imclose(bw, se);
    bw    = imopen(bw, se);

    cc    = bwconncomp(bw);
    stats = regionprops(cc, 'Area', 'BoundingBox', 'PixelIdxList');
    mask  = false(size(bw));
    tool_center_x = floor(w / 2);
    tool_w = 0;

    if ~isempty(stats)
        areas    = [stats.Area];
        min_area = max(500, 0.01 * h * w);
        [max_area, li] = max(areas);
        if max_area > min_area
            mask(stats(li).PixelIdxList) = true;
            bb     = stats(li).BoundingBox;
            tool_w = bb(3);
            tool_center_x = round(bb(1) + bb(3)/2);
        end
    end
    if tool_w == 0, return; end

    cfg.RANSAC_ITERATIONS   = 40;
    cfg.RANSAC_TOLERANCE    = max(1.2, tool_w * 0.006);
    cfg.RANSAC_CENTER_DELTA = floor(tool_w * 0.02);
    cfg.RANSAC_CENTER_GRID  = 5;
    cfg.ROI_MARGIN          = floor(tool_w * 0.05);

    mask = imfill(mask, 'holes');
    masked_gray = img_gray;
    masked_gray(~mask) = 0;
    tool_center_x = max(1, min(tool_center_x, w));

    img_left_raw  = masked_gray(:, 1:tool_center_x);
    img_right_raw = masked_gray(:, (tool_center_x+1):end);

    left_canny = edge(img_left_raw, 'Canny', [0.2, 0.65]);
    mask_left  = mask(:, 1:tool_center_x);
    try
        [R_L, xc_L, yc_L] = single_side(left_canny, mask_left, cfg, tool_w);
    catch; end

    img_rf         = flip(flip(img_right_raw, 2), 1);
    mask_rf        = flip(flip(mask(:, (tool_center_x+1):end), 2), 1);
    right_canny    = edge(img_rf, 'Canny', [0.2, 0.65]);
    try
        [R_R_f, xc_R_f, yc_R_f] = single_side(right_canny, mask_rf, cfg, tool_w);
        if ~isnan(R_R_f)
            w_r  = size(img_right_raw, 2);
            yc_R = h - yc_R_f + 1;
            xc_R = tool_center_x + (w_r - xc_R_f + 1);
            R_R  = R_R_f;
        end
    catch; end
end

function [R, xc, yc] = single_side(img_a, side_mask, cfg, tw)
    R = NaN; xc = NaN; yc = NaN;
    [all_y, all_x] = find(img_a > 0);
    if isempty(all_x), return; end

    anchorPt  = refined_anchor(all_x, all_y, cfg.ANGLE_REF, tw);
    y_center  = mean(all_y);
    top_idx   = find(all_y < y_center);
    if isempty(top_idx), top_idx = 1:length(all_y); end
    ty = all_y(top_idx); tx = all_x(top_idx);
    min_y = min(ty);
    on_top = find(ty <= (min_y + tw * 0.03));
    if isempty(on_top)
        [~, mi] = min(ty); topLeftPt = [tx(mi), ty(mi)];
    else
        [~, li] = min(tx(on_top)); ai = on_top(li);
        topLeftPt = [tx(ai), ty(ai)];
    end
    if topLeftPt(2) > anchorPt(2), topLeftPt = anchorPt; end
    if norm(topLeftPt - anchorPt) > tw * 0.07, topLeftPt = anchorPt; end

    rm  = cfg.ROI_MARGIN;
    roi = (all_x >= min(anchorPt(1),topLeftPt(1))-rm) & ...
          (all_x <= max(anchorPt(1),topLeftPt(1))+rm) & ...
          (all_y >= min(anchorPt(2),topLeftPt(2))-rm) & ...
          (all_y <= max(anchorPt(2),topLeftPt(2))+rm);
    xs_f = all_x(roi); ys_f = all_y(roi);
    if length(xs_f) < 3, return; end

    pts = [xs_f(:), ys_f(:)]; n = size(pts,1);
    dom = false(n,1);
    for i = 1:n
        for j = 1:n
            if i==j, continue; end
            if pts(j,1)<=pts(i,1) && pts(j,2)<=pts(i,2) && ...
               (pts(j,1)<pts(i,1) || pts(j,2)<pts(i,2))
                dom(i) = true; break;
            end
        end
    end
    xs = pts(~dom,1); ys = pts(~dom,2);
    if length(xs) < 2, return; end

    [xs_fit, ys_fit, xc, yc] = ransac_filt(xs, ys, topLeftPt(1), anchorPt(2), side_mask, cfg);
    if length(xs_fit) < 2, xs_fit = xs; ys_fit = ys; end
    R = circ_r(xs_fit, ys_fit, xc, yc);
end

function P = refined_anchor(all_x, all_y, ang, tw)
    li = find(all_x <= min(all_x) + tw*0.01);
    [~, ti] = min(all_y(li));
    si = li(ti);
    xs = all_x(si); ys = all_y(si);
    m  = -tand(ang); A = m; B = -1; C = ys - m*xs;
    d  = abs(A*all_x + B*all_y + C) / sqrt(A^2+B^2);
    bi = find(d <= 3);
    if isempty(bi), P = [xs, ys]; return; end
    xb = all_x(bi); yb = all_y(bi);
    [~, mi] = min(yb);
    P = [xb(mi), yb(mi)];
end

function [xsf, ysf, bxc, byc] = ransac_filt(xs, ys, xc0, yc0, smask, cfg)
    n = length(xs); [mh, mw] = size(smask);
    bxc = xc0; byc = yc0; xsf = []; ysf = [];
    if n < 2, return; end
    d  = cfg.RANSAC_CENTER_DELTA; gs = cfg.RANSAC_CENTER_GRID;
    cr = linspace(0, d, gs);
    [dx, dy] = meshgrid(cr, cr);
    cxs = xc0+dx(:); cys = yc0+dy(:);
    best = 0; bm = false(n,1);
    for c = 1:length(cxs)
        ix = round(cxs(c)); iy = round(cys(c));
        if ix<1||ix>mw||iy<1||iy>mh, continue; end
        if smask(iy,ix)==0, continue; end
        [im, ic] = ransac_fc(xs, ys, cxs(c), cys(c), cfg.RANSAC_TOLERANCE, cfg.RANSAC_ITERATIONS);
        if ic > best, best=ic; bm=im; bxc=cxs(c); byc=cys(c); end
    end
    if best > 1, xsf=xs(bm); ysf=ys(bm); end
end

function [im, cnt] = ransac_fc(xs, ys, xc, yc, tol, iters)
    n = length(xs); best = 0; im = false(n,1);
    for i = 1:iters
        si = randi(n);
        Rh = sqrt((xs(si)-xc)^2+(ys(si)-yc)^2);
        d  = sqrt((xs-xc).^2+(ys-yc).^2);
        cm = abs(d-Rh) <= tol;
        if sum(cm) > best, best=sum(cm); im=cm; end
    end
    cnt = best;
end

function R = circ_r(x, y, xc, yc)
    x=x(:); y=y(:);
    R = sqrt(sum((x-xc).^2+(y-yc).^2)/length(x));
end
