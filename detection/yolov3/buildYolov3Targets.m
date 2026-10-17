function targets = buildYolov3Targets(decoded, groundTruth, inputSize, anchorBoxes, anchorBoxMasks, numClasses, ignoreThreshold)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% buildYolov3Targets  Training targets and masks for every head.
%   decoded        : 1-by-K cell, decoded{k} = decodeHeadBoxes output for head k
%                    (h_k-by-w_k-by-4-by-A_k-by-B), used for the ignore rule
%   groundTruth    : B-by-1 cell, row n = [x y w h classIndex] boxes of image n
%                    in network-input pixels (see createBatchData)
%   inputSize      : [height width] of the network input
%   anchorBoxes    : N-by-2 [height width] anchors; anchorBoxMasks{k} lists the
%                    rows of anchorBoxes that head k predicts
%   numClasses     : number of classes C
%   ignoreThreshold: predictions whose box overlaps any ground truth by more
%                    than this IoU are not penalized as background (paper, Sec. 2.1)
%
%   targets(k) fields, all h-by-w-by-*-by-A-by-B single:
%     xy       (2 ch) cell-relative center, the target of sigmoid(tx), sigmoid(ty)
%     wh       (2 ch) log(box / anchor), the target of tw, th
%     cls      (C ch) one-hot class target (independent logistic classifiers)
%     positive (1 ch) 1 where an anchor is responsible for a ground-truth box
%     negative (1 ch) 1 where the objectness target is 0 (not positive, not ignored)

numHeads = numel(decoded);
batchSize = numel(groundTruth);

% Which head and which slot inside that head own each anchor row.
headOf = zeros(size(anchorBoxes, 1), 1);
slotOf = zeros(size(anchorBoxes, 1), 1);
for k = 1:numHeads
    headOf(anchorBoxMasks{k}) = k;
    slotOf(anchorBoxMasks{k}) = 1:numel(anchorBoxMasks{k});
end

targets = struct('xy', {}, 'wh', {}, 'cls', {}, 'positive', {}, 'negative', {});
ignore = cell(numHeads, 1);
gridSize = zeros(numHeads, 3);                        % [h w A]
for k = 1:numHeads
    [h, w, ~, A, ~] = size(decoded{k});
    gridSize(k, :) = [h w A];
    targets(k).xy = zeros(h, w, 2, A, batchSize, 'single');
    targets(k).wh = zeros(h, w, 2, A, batchSize, 'single');
    targets(k).cls = zeros(h, w, numClasses, A, batchSize, 'single');
    targets(k).positive = zeros(h, w, 1, A, batchSize, 'single');
    ignore{k} = false(h, w, 1, A, batchSize);
end

for n = 1:batchSize
    gt = groundTruth{n};
    if isempty(gt)
        continue
    end
    gtBoxes = gt(:, 1:4);
    gtClass = gt(:, 5);

    % Ignore rule: a prediction that already overlaps a real object well is
    % neither rewarded nor penalized for objectness.
    for k = 1:numHeads
        h = gridSize(k, 1); w = gridSize(k, 2); A = gridSize(k, 3);
        predBoxes = reshape(permute(decoded{k}(:, :, :, :, n), [1 2 4 3]), [], 4);   % (h*w*A)-by-4
        bestOverlap = max(bboxOverlapRatio(predBoxes, gtBoxes), [], 2);
        ignore{k}(:, :, 1, :, n) = reshape(bestOverlap > ignoreThreshold, h, w, 1, A);
    end

    % Each ground-truth box is assigned to the one anchor whose shape fits it
    % best (IoU of the two boxes placed on a common center).
    for m = 1:size(gtBoxes, 1)
        gw = gtBoxes(m, 3);
        gh = gtBoxes(m, 4);
        if gw <= 0 || gh <= 0 || gtClass(m) < 1 || gtClass(m) > numClasses
            continue
        end
        overlap = min(gw, anchorBoxes(:, 2)) .* min(gh, anchorBoxes(:, 1));
        shapeIoU = overlap ./ (gw * gh + anchorBoxes(:, 1) .* anchorBoxes(:, 2) - overlap);
        [~, best] = max(shapeIoU);
        k = headOf(best);
        if k == 0
            continue                                  % anchor not used by any head
        end
        a = slotOf(best);
        h = gridSize(k, 1); w = gridSize(k, 2);

        % Same coordinate convention as decodeHeadBoxes: u = pixel - 1.
        gridX = (gtBoxes(m, 1) - 1 + gw / 2) / (inputSize(2) / w);
        gridY = (gtBoxes(m, 2) - 1 + gh / 2) / (inputSize(1) / h);
        col = min(max(floor(gridX), 0), w - 1);
        row = min(max(floor(gridY), 0), h - 1);

        targets(k).xy(row + 1, col + 1, :, a, n) = reshape([gridX - col, gridY - row], 1, 1, 2);
        targets(k).wh(row + 1, col + 1, :, a, n) = reshape(log([gw / anchorBoxes(best, 2), gh / anchorBoxes(best, 1)]), 1, 1, 2);
        targets(k).cls(row + 1, col + 1, :, a, n) = 0;
        targets(k).cls(row + 1, col + 1, gtClass(m), a, n) = 1;
        targets(k).positive(row + 1, col + 1, 1, a, n) = 1;
    end
end

for k = 1:numHeads
    targets(k).negative = single(~targets(k).positive & ~ignore{k});
end
end
