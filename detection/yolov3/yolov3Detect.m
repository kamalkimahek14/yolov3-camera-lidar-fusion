function [bboxes, scores, labels] = yolov3Detect(net, X, networkOutputs, anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% yolov3Detect  Detect objects in one image with a trained YOLO v3 dlnetwork.
%   X                  : h-by-w-by-3-by-1 dlarray, format 'SSCB', pixels in [0, 1]
%                        (see preprocessData); may hold a gpuArray
%   networkOutputs     : names of the head output layers
%   anchorBoxes        : N-by-2 [height width] anchors; anchorBoxMasks{k} = rows of head k
%   confidenceThreshold: keep boxes whose objectness x class probability exceeds this
%   overlapThreshold   : IoU used by non-maximum suppression
%   classNames         : cell array of class names, in training order
%   bboxes             : M-by-4 [x y w h] in network-input pixels
%   scores             : M-by-1 objectness x class probability
%   labels             : M-by-1 categorical with categories classNames
%
% The score of a box is sigmoid(to) * sigmoid(class logit), the best class per
% anchor is kept, and overlapping boxes of the same class are removed with
% selectStrongestBboxMulticlass.

assert(size(X, 4) == 1, 'yolov3Detect:batch', 'Pass one image at a time.');
inputSize = size(X, [1 2]);
classNames = cellstr(classNames);

numHeads = numel(networkOutputs);
heads = cell(numHeads, 1);
[heads{:}] = predict(net, X, 'Outputs', networkOutputs);

allBoxes = zeros(0, 4);
allScores = zeros(0, 1);
allClasses = zeros(0, 1);
for k = 1:numHeads
    anchors = anchorBoxes(anchorBoxMasks{k}, :);
    [t, cls] = splitHeadOutput(heads{k}, size(anchors, 1));
    t = double(gather(extractdata(t)));           % h-by-w-by-5-by-A
    cls = double(gather(extractdata(cls)));       % h-by-w-by-C-by-A

    boxes = decodeHeadBoxes(t, anchors, inputSize);                 % h-by-w-by-4-by-A
    classProb = logistic(cls) .* logistic(t(:, :, 5, :));          % h-by-w-by-C-by-A
    [bestProb, bestClass] = max(classProb, [], 3);                 % h-by-w-by-1-by-A

    keep = bestProb(:) > confidenceThreshold;
    boxes = reshape(permute(boxes, [1 2 4 3]), [], 4);              % same order as bestProb(:)
    allBoxes = [allBoxes; boxes(keep, :)]; %#ok<AGROW>
    allScores = [allScores; bestProb(keep)]; %#ok<AGROW>
    allClasses = [allClasses; bestClass(keep)]; %#ok<AGROW>
end

% Clip to the image and drop boxes that end up empty.
x1 = max(allBoxes(:, 1), 1);
y1 = max(allBoxes(:, 2), 1);
x2 = min(allBoxes(:, 1) + allBoxes(:, 3), inputSize(2) + 1);
y2 = min(allBoxes(:, 2) + allBoxes(:, 4), inputSize(1) + 1);
allBoxes = [x1, y1, x2 - x1, y2 - y1];
valid = allBoxes(:, 3) > 0 & allBoxes(:, 4) > 0;

bboxes = allBoxes(valid, :);
scores = allScores(valid);
labels = categorical(allClasses(valid), 1:numel(classNames), classNames);

if ~isempty(bboxes)
    [bboxes, scores, labels] = selectStrongestBboxMulticlass(bboxes, scores, labels, ...
        'RatioType', 'Union', 'OverlapThreshold', overlapThreshold);
end
end

function y = logistic(x)
y = 1 ./ (1 + exp(-x));
end
