function data = augmentData(data)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% augmentData  Random photometric and mirror augmentation of training rows.
%   data : datastore rows {image, boxes, labels}, boxes as [x y w h] pixels
% For each row:
%   - color images get a random hue, saturation and brightness shift
%     (jitterColorHSV), which helps with the lab's changing light;
%   - half of the rows are mirrored left-right, and their boxes with them.
% Labels are unchanged; no box is lost, so no row needs to be dropped.

for r = 1:size(data, 1)
    I = data{r, 1};
    boxes = data{r, 2};

    if size(I, 3) == 3
        I = jitterColorHSV(I, 'Hue', 0.05, 'Saturation', 0.15, 'Brightness', 0.15);
    end

    if rand < 0.5
        width = size(I, 2);
        I = flip(I, 2);
        % A box spanning u = x-1 .. x-1+w maps to width-u, so its new left pixel is width+2-x-w.
        boxes(:, 1) = width + 2 - boxes(:, 1) - boxes(:, 3);
    end

    data(r, 1:2) = {I, boxes};
end
end
