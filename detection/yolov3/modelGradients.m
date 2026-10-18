function [gradients, loss, state] = modelGradients(net, X, groundTruth, anchorBoxes, anchorBoxMasks, ignoreThreshold, networkOutputs)
% YOLO v3 (Redmon & Farhadi, 2018). Copyright (c) 2026 Kamal Mehta. MIT License.
%
% modelGradients  YOLO v3 loss and its gradients for one mini-batch.
% Call it through dlfeval:
%   [gradients, loss, state] = dlfeval(@modelGradients, net, X, groundTruth, ...
%       anchorBoxes, anchorBoxMasks, ignoreThreshold, networkOutputs)
%   net            : dlnetwork with one output layer per detection head
%   X              : h-by-w-by-3-by-B dlarray, format 'SSCB', pixels in [0, 1]
%   groundTruth    : B-by-1 cell of [x y w h classIndex] rows (createBatchData)
%   anchorBoxes    : N-by-2 [height width] anchors; anchorBoxMasks{k} = rows used by head k
%   ignoreThreshold: IoU above which a non-responsible prediction is not
%                    treated as background
%   networkOutputs : names of the head output layers, one per anchorBoxMasks cell
%   loss           : struct with totalLoss, boxLoss, objLoss, clsLoss (dlarray scalars)
%
% Loss terms, following the paper (Sec. 2.1 and 2.2):
%   box       : sum of squared errors on sigmoid(tx), sigmoid(ty), tw, th for
%               responsible anchors only
%   objectness: binary cross-entropy, target 1 for responsible anchors and 0
%               for background; ignored anchors add nothing
%   class     : binary cross-entropy per class (independent logistic
%               classifiers) for responsible anchors only
% All three are summed over the batch and divided by the batch size.

numHeads = numel(networkOutputs);
heads = cell(numHeads, 1);
[heads{:}, state] = forward(net, X, 'Outputs', networkOutputs);

inputSize = size(X, [1 2]);
batchSize = size(X, 4);

boxTerms = cell(numHeads, 1);
classLogits = cell(numHeads, 1);
decoded = cell(numHeads, 1);
for k = 1:numHeads
    anchors = anchorBoxes(anchorBoxMasks{k}, :);
    [boxTerms{k}, classLogits{k}] = splitHeadOutput(heads{k}, size(anchors, 1));
    decoded{k} = decodeHeadBoxes(double(gather(extractdata(boxTerms{k}))), anchors, inputSize);
end
numClasses = size(classLogits{1}, 3);

% Targets are plain numeric arrays: no gradient flows through them.
targets = buildYolov3Targets(decoded, groundTruth, inputSize, anchorBoxes, anchorBoxMasks, numClasses, ignoreThreshold);

boxLoss = 0;
objLoss = 0;
clsLoss = 0;
for k = 1:numHeads
    t = boxTerms{k};
    pos = targets(k).positive;
    neg = targets(k).negative;

    centerErr = (logistic(t(:, :, 1:2, :, :)) - targets(k).xy) .^ 2 .* pos;
    sizeErr = (t(:, :, 3:4, :, :) - targets(k).wh) .^ 2 .* pos;
    boxLoss = boxLoss + sum(centerErr(:)) + sum(sizeErr(:));

    objLogit = t(:, :, 5, :, :);
    objErr = crossEntropyWithLogits(objLogit, 1) .* pos + crossEntropyWithLogits(objLogit, 0) .* neg;
    objLoss = objLoss + sum(objErr(:));

    clsErr = crossEntropyWithLogits(classLogits{k}, targets(k).cls) .* pos;
    clsLoss = clsLoss + sum(clsErr(:));
end

loss.boxLoss = boxLoss / batchSize;
loss.objLoss = objLoss / batchSize;
loss.clsLoss = clsLoss / batchSize;
loss.totalLoss = loss.boxLoss + loss.objLoss + loss.clsLoss;

gradients = dlgradient(loss.totalLoss, net.Learnables);
end

function y = logistic(x)
y = 1 ./ (1 + exp(-x));
end

function e = crossEntropyWithLogits(z, target)
% Binary cross-entropy of sigmoid(z) against target, written in the
% numerically stable form max(z,0) - z*target + log(1 + exp(-|z|)).
e = max(z, 0) - z .* target + log(1 + exp(-abs(z)));
end
