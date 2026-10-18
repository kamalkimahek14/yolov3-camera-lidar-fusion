% YOLO v3 (Redmon & Farhadi, 2018). Copyright (c) 2026 Kamal Mehta. MIT License.
%
% test_yolov3_helpers  Self-checks for the helpers that need no trained network.
% Run from the repository root after addpath(genpath(pwd)). Needs Image
% Processing Toolbox (imresize).

%% Learning-rate schedule matches the 2021 training log
assert(abs(piecewiseLearningRateWithWarmup(1, 1e-4, 500, 11000) - 1.6e-15) < 1e-20);
assert(piecewiseLearningRateWithWarmup(500, 1e-4, 500, 11000) == 1e-4);
assert(piecewiseLearningRateWithWarmup(6799, 1e-4, 500, 11000) == 1e-4);
assert(abs(piecewiseLearningRateWithWarmup(6800, 1e-4, 500, 11000) - 1e-5) < 1e-18);
assert(abs(piecewiseLearningRateWithWarmup(9950, 1e-4, 500, 11000) - 1e-6) < 1e-18);

%% Box overlap (intersection over union)
assert(abs(boxIoU([0 0 10 10], [5 0 10 10]) - 50/150) < 1e-12);
assert(boxIoU([0 0 10 10], [20 20 5 5]) == 0);
assert(isequal(size(boxIoU(rand(3,4)+[0 0 1 1], rand(2,4)+[0 0 1 1])), [3 2]));

%% Box encoding (targets) and decoding agree
inputSize = [227 227];
anchorBoxes = [120 90; 60 40; 30 20; 12 10];      % [height width], largest first
anchorBoxMasks = {[1 2], [3 4]};
grids = {[14 14], [28 28]};
numClasses = 4;
decoded = cell(1, 2);
for k = 1:2
    decoded{k} = ones(grids{k}(1), grids{k}(2), 4, numel(anchorBoxMasks{k}), 1);   % 1-pixel boxes at (1,1)
end
gtBox = [101.3 57.8 55 64];                       % fits anchor 2 ([60 40]) best
targets = buildYolov3Targets(decoded, {[gtBox 3]}, inputSize, anchorBoxes, anchorBoxMasks, numClasses, 0.5);

assert(nnz(targets(1).positive) == 1 && nnz(targets(2).positive) == 0, 'one responsible anchor, on head 1');
[row, col, ~] = ind2sub(size(targets(1).positive), find(targets(1).positive));
slot = 2;                                          % anchor 2 is slot 2 of head 1
assert(targets(1).positive(row, col, 1, slot) == 1);
assert(isequal(squeeze(targets(1).cls(row, col, :, slot))', [0 0 1 0]));

% Build raw terms that hit the targets exactly, decode them, compare.
t = zeros(14, 14, 5, 2);
xy = squeeze(targets(1).xy(row, col, :, slot));
t(row, col, 1:2, slot) = log(xy ./ (1 - xy));      % inverse of the logistic function
t(row, col, 3:4, slot) = targets(1).wh(row, col, :, slot);
boxes = decodeHeadBoxes(t, anchorBoxes(anchorBoxMasks{1}, :), inputSize);
decodedBox = squeeze(boxes(row, col, :, slot))';
assert(max(abs(decodedBox - gtBox)) < 1e-3, 'decode(encode(box)) must return the box');

%% Ignore rule: a prediction that overlaps the object is not background
decoded{1}(row, col, :, 1) = reshape(gtBox, 1, 1, 4);   % slot 1 predicts the same box
targets = buildYolov3Targets(decoded, {[gtBox 3]}, inputSize, anchorBoxes, anchorBoxMasks, numClasses, 0.5);
assert(targets(1).negative(row, col, 1, 1) == 0, 'overlapping prediction is ignored');
assert(targets(1).negative(1, 1, 1, 1) == 1, 'empty cell stays background');

%% Preprocessing rescales boxes with the image
row1 = preprocessData({zeros(100, 200, 3, 'uint8'), [1 1 200 100], categorical("Box")}, [227 227 3]);
assert(isequal(size(row1{1}), [227 227 3]) && isa(row1{1}, 'single'));
assert(max(abs(row1{2} - [1 1 227 227])) < 1e-9, 'full-image box stays full-image');

%% Mirroring keeps a full-image box in place
rng(1);
for trial = 1:4
    out = augmentData({zeros(50, 80, 3, 'uint8'), [1 1 80 50], categorical("Box")});
    assert(isequal(out{2}, [1 1 80 50]));
end

disp('All detection/yolov3 self-checks passed.');
