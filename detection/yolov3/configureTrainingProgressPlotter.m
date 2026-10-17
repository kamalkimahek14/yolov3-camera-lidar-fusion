function [lossLine, rateLine] = configureTrainingProgressPlotter(fig)
% Written 2026 by Kamal Mehta as an independent implementation of YOLO v3 (Redmon & Farhadi, 2018); MIT.
%
% configureTrainingProgressPlotter  Live plots of total loss and learning rate.
%   fig      : figure handle to draw in (it is cleared)
%   lossLine : animatedline for the total loss (top panel)
%   rateLine : animatedline for the learning rate (bottom panel)
% Add points with addpoints(lossLine, iteration, value).

clf(fig);
lossAxes = subplot(2, 1, 1, 'Parent', fig);
lossLine = animatedline(lossAxes, 'Color', [0.85 0.33 0.10]);
xlabel(lossAxes, 'Iteration');
ylabel(lossAxes, 'Total loss');
grid(lossAxes, 'on');

rateAxes = subplot(2, 1, 2, 'Parent', fig);
rateLine = animatedline(rateAxes, 'Color', [0 0.45 0.74]);
xlabel(rateAxes, 'Iteration');
ylabel(rateAxes, 'Learning rate');
grid(rateAxes, 'on');
end
