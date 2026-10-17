function lgraph = addSecondDetectionHead(lgraph, anchorMask, numPredictorsPerAnchor)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% addSecondDetectionHead  Fine-grid head (small anchors), added unconnected.
%   anchorMask             : rows of anchorBoxes this head predicts
%   numPredictorsPerAnchor : 5 + number of classes
% As in the paper (Sec. 2.3), coarse features are upsampled 2x and
% concatenated with an earlier, finer feature map before predicting:
%   upsample1Detection2 (2x nearest) -> depthConcat1Detection2/in1
%   'fire5-concat'                   -> depthConcat1Detection2/in2 (connected by the script)
%   -> conv1Detection2 (3x3, 128) -> relu1Detection2 -> conv2Detection2 (1x1)
% resize2dLayer needs Image Processing Toolbox R2020b or later.

numOutputs = numel(anchorMask) * numPredictorsPerAnchor;
head = [
    resize2dLayer('Scale', 2, 'Method', 'nearest', 'Name', 'upsample1Detection2')
    depthConcatenationLayer(2, 'Name', 'depthConcat1Detection2')
    convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv1Detection2', 'WeightsInitializer', 'he')
    reluLayer('Name', 'relu1Detection2')
    convolution2dLayer(1, numOutputs, 'Padding', 'same', 'Name', 'conv2Detection2', 'WeightsInitializer', 'he')
    ];
lgraph = addLayers(lgraph, head);
end
