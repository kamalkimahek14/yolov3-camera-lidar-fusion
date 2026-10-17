function [t, cls] = splitHeadOutput(out, numAnchors)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% splitHeadOutput  Split one detection-head output into box terms and class logits.
%   out : h-by-w-by-(numAnchors*P)-by-B array (formatted 'SSCB' dlarray, plain
%         dlarray or numeric), with P = 5 + numClasses channels per anchor.
%         Channel layout used throughout detection/yolov3/: for anchor a,
%         channels (a-1)*P + (1:P) hold [tx ty tw th to class_1 ... class_C].
%   t   : h-by-w-by-5-by-numAnchors-by-B, raw [tx ty tw th to]
%   cls : h-by-w-by-C-by-numAnchors-by-B, raw class logits

if isa(out, 'dlarray')
    out = stripdims(out);           % plain dlarray keeps autodiff tracing
end
[h, w, ch, b] = size(out);          % b = 1 when the batch has one image
P = ch / numAnchors;
assert(P == round(P) && P > 5, 'splitHeadOutput:channels', ...
    'Head has %d channels, not a multiple of %d anchors with 5+C terms each.', ch, numAnchors);

r = reshape(out, h, w, P, numAnchors, b);
t = r(:, :, 1:5, :, :);
cls = r(:, :, 6:end, :, :);
end
