function [X, groundTruth] = createBatchData(data, classNames)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% createBatchData  Stack one mini-batch for the custom training loop.
%   data        : B-by-3 cell of preprocessed rows {image, boxes, labels}
%   classNames  : class names in training order
%   X           : h-by-w-by-3-by-B numeric array (wrap it as dlarray(X, 'SSCB'))
%   groundTruth : B-by-1 cell; row n holds [x y w h classIndex] for image n,
%                 where classIndex is the position of the label in classNames

X = cat(4, data{:, 1});
groundTruth = cell(size(data, 1), 1);
names = cellstr(classNames);
for n = 1:size(data, 1)
    [~, classIndex] = ismember(cellstr(data{n, 3}), names);
    rows = [double(data{n, 2}), classIndex(:)];
    groundTruth{n} = rows(classIndex(:) > 0, :);
end
end
