function lgraph = squeezenetFeatureExtractor(net, inputSize)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% squeezenetFeatureExtractor  Backbone for the detector: SqueezeNet up to 'fire9-concat'.
%   net       : pretrained network from squeezenet()
%   inputSize : network input size, e.g. [227 227 3]
%   lgraph    : layer graph that keeps every layer feeding 'fire9-concat',
%               drops the classification tail, and takes inputs in [0, 1]
%               (no mean subtraction), which matches preprocessData.
% The detection heads attach to 'fire9-concat' (coarse grid) and
% 'fire5-concat' (fine grid); see addFirstDetectionHead / addSecondDetectionHead.

lgraph = layerGraph(net);
keepUpTo = 'fire9-concat';

% Walk the connections backwards from keepUpTo to find every layer it needs.
src = cellfun(@(s) strtok(s, '/'), lgraph.Connections.Source, 'UniformOutput', false);
dst = cellfun(@(s) strtok(s, '/'), lgraph.Connections.Destination, 'UniformOutput', false);
needed = {keepUpTo};
frontier = {keepUpTo};
while ~isempty(frontier)
    parents = src(ismember(dst, frontier));
    frontier = setdiff(unique(parents), needed);
    needed = [needed; frontier(:)]; %#ok<AGROW>
end
allNames = {lgraph.Layers.Name}';
lgraph = removeLayers(lgraph, setdiff(allNames, needed));

% Replace the input layer so the network reads [0, 1] pixels directly.
isInput = arrayfun(@(L) isa(L, 'nnet.cnn.layer.ImageInputLayer'), lgraph.Layers);
inputName = lgraph.Layers(isInput).Name;
lgraph = replaceLayer(lgraph, inputName, ...
    imageInputLayer(inputSize, 'Normalization', 'none', 'Name', inputName));
end
