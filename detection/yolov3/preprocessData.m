function data = preprocessData(data, targetSize)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% preprocessData  Resize images to the network input and scale pixels to [0, 1].
% Two call forms, both used by the thesis scripts:
%   I = preprocessData(I, targetSize)
%       one frame (live scripts): returns a single-precision image
%   data = preprocessData(data, targetSize)
%       datastore rows {image, boxes, labels} (training): resizes each image
%       and rescales its [x y w h] boxes to match
% Grayscale frames are repeated into three channels.

if ~iscell(data)
    data = resizeFrame(data, targetSize);
    return
end

for r = 1:size(data, 1)
    I = data{r, 1};
    boxes = data{r, 2};
    sy = targetSize(1) / size(I, 1);
    sx = targetSize(2) / size(I, 2);
    % Scale box edges about the image corner (u = pixel - 1), then back to pixels.
    boxes = [(boxes(:, 1) - 1) * sx + 1, (boxes(:, 2) - 1) * sy + 1, boxes(:, 3) * sx, boxes(:, 4) * sy];
    data(r, 1:2) = {resizeFrame(I, targetSize), boxes};
end
end

function I = resizeFrame(I, targetSize)
if size(I, 3) == 1
    I = repmat(I, 1, 1, 3);
end
I = im2single(imresize(I, targetSize(1:2)));
end
