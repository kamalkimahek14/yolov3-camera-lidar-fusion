% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% smoke_test_training_step  Builds the YOLO v3 network the way train_yolov3 does,
% then runs five training steps on random images with made-up boxes. It needs
% no dataset; it checks that the network builds, the loss is finite, the
% gradients flow, and the loss falls when the same batch is repeated.
% Needs Deep Learning Toolbox, its SqueezeNet support package and Image
% Processing Toolbox. Run from the repository root after addpath(genpath(pwd)).

rng(0);
networkInputSize = [227 227 3];
classNames = {'Cylinder', 'Box', 'Pyramid', 'Traffic_cone'};
numClasses = numel(classNames);

% 24 anchors as [height width], largest area first, like train_yolov3
heights = round(linspace(160, 10, 24))';
anchorBoxes = [heights, round(heights .* 0.8)];
anchorBoxMasks = {1:12, 13:24};

lgraph = squeezenetFeatureExtractor(squeezenet, networkInputSize);
numPredictorsPerAnchor = 5 + numClasses;
lgraph = addFirstDetectionHead(lgraph, anchorBoxMasks{1}, numPredictorsPerAnchor);
lgraph = addSecondDetectionHead(lgraph, anchorBoxMasks{2}, numPredictorsPerAnchor);
lgraph = connectLayers(lgraph, 'fire9-concat', 'conv1Detection1');
lgraph = connectLayers(lgraph, 'relu1Detection1', 'upsample1Detection2');
lgraph = connectLayers(lgraph, 'fire5-concat', 'depthConcat1Detection2/in2');
networkOutputs = ["conv2Detection1"; "conv2Detection2"];
net = dlnetwork(lgraph);

% Two random images with boxes [x y w h classIndex]
X = dlarray(single(rand([networkInputSize 2])), 'SSCB');
groundTruth = {[50 60 40 50 1]; [100 80 30 30 3; 20 20 25 40 2]};

velocity = [];
% A small fixed learning rate stands in for the warm-up of train_yolov3: at the
% full rate of 1e-4 an untrained network takes too large a first step.
learnRate = 1e-6;
losses = zeros(1, 5);
for step = 1:5
    [gradients, loss, state] = dlfeval(@modelGradients, net, X, groundTruth, ...
        anchorBoxes, anchorBoxMasks, 0.5, networkOutputs);
    losses(step) = double(extractdata(loss.totalLoss));
    assert(isfinite(losses(step)), 'loss must be finite');
    g = gradients.Value;
    assert(all(cellfun(@(v) all(isfinite(extractdata(v)), 'all'), g)), 'gradients must be finite');
    assert(any(cellfun(@(v) any(extractdata(v) ~= 0, 'all'), g)), 'gradients must not all be zero');
    [net, velocity] = sgdmupdate(net, gradients, velocity, learnRate);
    net.State = state;
end
fprintf('Training-step losses on one repeated batch: %s\n', mat2str(losses, 5));
assert(all(diff(losses) < 0) && losses(end) < losses(1) / 10, 'the loss should fall steadily on a repeated batch');

% Detection runs too when Computer Vision Toolbox is present (non-maximum suppression)
if exist('selectStrongestBboxMulticlass', 'file')
    [bboxes, scores, labels] = yolov3Detect(net, X(:, :, :, 1), networkOutputs, ...
        anchorBoxes, anchorBoxMasks, 0.01, 0.5, classNames);
    fprintf('Detection ran: %d boxes before training on real data.\n', size(bboxes, 1));
end
disp('Smoke test passed: the network builds, trains for five steps, and the loss falls.');
