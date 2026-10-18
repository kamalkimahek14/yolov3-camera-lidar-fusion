function lr = piecewiseLearningRateWithWarmup(iteration, baseRate, warmupPeriod, numIterations)
% YOLO v3 (Redmon & Farhadi, 2018). Copyright (c) 2026 Kamal Mehta. MIT License.
%
% piecewiseLearningRateWithWarmup  Learning rate for one iteration.
% Reproduces the schedule recorded in the 2021 training log of the thesis:
%   warm-up      : baseRate * (iteration / warmupPeriod)^4 up to warmupPeriod
%   then         : baseRate
%   after 60% of the post-warm-up iterations : baseRate / 10
%   after 90% of the post-warm-up iterations : baseRate / 100
% With the thesis settings (1e-4, 500, 11000): 1e-4 until 6,800, 1e-5 until 9,950,
% then 1e-6 (see docs/lr-schedule.svg).

if iteration <= warmupPeriod
    lr = baseRate * (iteration / warmupPeriod) ^ 4;
    return
end
decayStart = warmupPeriod + floor(0.6 * (numIterations - warmupPeriod));
finalStart = warmupPeriod + floor(0.9 * (numIterations - warmupPeriod));
if iteration < decayStart
    lr = baseRate;
elseif iteration < finalStart
    lr = baseRate / 10;
else
    lr = baseRate / 100;
end
end
