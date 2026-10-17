% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% boxIoU  Pairwise intersection over union of axis-aligned boxes.
%   iou = boxIoU(A, B) returns an M-by-N matrix for M boxes in A and N boxes in B,
%   each given as [x y width height]. Same result as the union ratio of
%   bboxOverlapRatio, without needing Computer Vision Toolbox.
function iou = boxIoU(A, B)
    if isempty(A) || isempty(B)
        iou = zeros(size(A, 1), size(B, 1));
        return
    end
    ax2 = A(:, 1) + A(:, 3);  ay2 = A(:, 2) + A(:, 4);
    bx2 = B(:, 1) + B(:, 3);  by2 = B(:, 2) + B(:, 4);
    w = max(0, min(ax2, bx2') - max(A(:, 1), B(:, 1)'));
    h = max(0, min(ay2, by2') - max(A(:, 2), B(:, 2)'));
    inter = w .* h;
    iou = inter ./ (A(:, 3) .* A(:, 4) + (B(:, 3) .* B(:, 4))' - inter);
end
