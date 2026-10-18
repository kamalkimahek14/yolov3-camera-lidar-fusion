function displayLossInfo(iteration, lr, loss)
% YOLO v3 (Redmon & Farhadi, 2018). Copyright (c) 2026 Kamal Mehta. MIT License.
%
% displayLossInfo  Print one line of training progress.
%   loss : struct from modelGradients (totalLoss, boxLoss, objLoss, clsLoss)

value = @(x) double(gather(extractdata(x)));
fprintf('Iteration %6d | LR %.3g | total %.4f | box %.4f | objectness %.4f | class %.4f\n', ...
    iteration, lr, value(loss.totalLoss), value(loss.boxLoss), value(loss.objLoss), value(loss.clsLoss));
end
