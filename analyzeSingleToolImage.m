function [edge_radius_px, tool_length_px, worn_area_px, ideal_worn_area_px] = analyzeSingleToolImage(imgPath)
% Analyze a single worn tool image using the hardcoded fresh reference.
%
% Input:
%   imgPath              – full path to the worn tool image (char or string)
% Outputs:
%   edge_radius_px       – mean RANSAC edge radius in pixels
%   tool_length_px       – leftmost-to-rightmost pixel distance
%   worn_area_px         – pixel count of worn region (fresh mask minus worn mask)
%   ideal_worn_area_px   – theoretical ideal worn area in pixels^2
%
% This is a thin wrapper around agent_worn_area so that Python callers
% only need to supply the single worn image path.

    imgPath = char(imgPath);
    freshImgPath = 'C:\Users\Bedirhan\Desktop\Main\Courses\Senior\IE490-491\IE491\TestData\Fresh_Unworn\tltest0102032026_110457 AM.jpg';

    [edge_radius_px, tool_length_px, worn_area_px, ideal_worn_area_px] = agent_worn_area(imgPath, freshImgPath);
end
