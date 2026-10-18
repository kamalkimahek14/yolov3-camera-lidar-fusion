function lgraph = addFirstDetectionHead(lgraph, anchorMask, numPredictorsPerAnchor)
% YOLO v3 (Redmon & Farhadi, 2018). Copyright (c) 2026 Kamal Mehta. MIT License.
%
% addFirstDetectionHead  Coarse-grid head (large anchors), added unconnected.
%   anchorMask             : rows of anchorBoxes this head predicts
%   numPredictorsPerAnchor : 5 + number of classes
% Layers: conv1Detection1 (3x3, 256) -> relu1Detection1 -> conv2Detection1
% (1x1, numel(anchorMask)*numPredictorsPerAnchor). The training script then
% connects 'fire9-concat' -> 'conv1Detection1'; 'relu1Detection1' also feeds
% the upsampling step of the second head (paper, Sec. 2.3).

numOutputs = numel(anchorMask) * numPredictorsPerAnchor;
head = [
    convolution2dLayer(3, 256, 'Padding', 'same', 'Name', 'conv1Detection1', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu1Detection1')
    convolution2dLayer(1, numOutputs, 'Padding', 'same', 'Name', 'conv2Detection1', 'WeightsInitializer', 'he')
    ];
lgraph = addLayers(lgraph, head);
end
