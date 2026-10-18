function boxes = decodeHeadBoxes(t, anchors, inputSize)
% YOLO v3 (Redmon & Farhadi, 2018). Copyright (c) 2026 Kamal Mehta. MIT License.
%
% decodeHeadBoxes  Turn raw box terms of one head into [x y w h] boxes in pixels.
%   t         : numeric h-by-w-by-5-by-A-by-B, raw [tx ty tw th to] (see splitHeadOutput)
%   anchors   : A-by-2 anchor sizes [height width] in input pixels, as returned by
%               estimateAnchorBoxes, in the order of this head's anchor mask
%   inputSize : [height width] of the network input
%   boxes     : h-by-w-by-4-by-A-by-B, MATLAB [x y w h] (x, y = upper-left pixel)
%
% Paper, Sec. 2.1: b_x = sigmoid(t_x) + c_x, b_y = sigmoid(t_y) + c_y,
% b_w = p_w * exp(t_w), b_h = p_h * exp(t_h), with (c_x, c_y) the cell offset.
% Coordinates here use u = pixel - 1, so the image spans u in [0, width];
% encodeGroundTruth in buildYolov3Targets uses the same convention.

[h, w, ~, A, ~] = size(t);
strideX = inputSize(2) / w;
strideY = inputSize(1) / h;
[cellX, cellY] = meshgrid(0:w-1, 0:h-1);          % h-by-w cell offsets

anchorW = reshape(anchors(:, 2), 1, 1, 1, A);
anchorH = reshape(anchors(:, 1), 1, 1, 1, A);

centerX = (logistic(t(:, :, 1, :, :)) + cellX) * strideX;
centerY = (logistic(t(:, :, 2, :, :)) + cellY) * strideY;
boxW = anchorW .* exp(min(max(t(:, :, 3, :, :), -10), 10));   % clamp keeps early-training boxes finite
boxH = anchorH .* exp(min(max(t(:, :, 4, :, :), -10), 10));

boxes = cat(3, centerX - boxW / 2 + 1, centerY - boxH / 2 + 1, boxW, boxH);
end

function y = logistic(x)
y = 1 ./ (1 + exp(-x));
end
