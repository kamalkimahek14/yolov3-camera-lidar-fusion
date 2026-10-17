function showReprojectionError(err)
% Written 2026 by Kamal Mehta; MIT.
%
% showReprojectionError  Plots the camera and LiDAR reprojection errors from the
% calibration scripts: one histogram per sensor, with the mean and one standard
% deviation marked.
%   err : struct with fields Lidar, Camera (error per target, metres),
%         meanLidar, meanCamera, SD_Lidar, SD_Camera

figure('Name', 'Reprojection error');
plotOne(1, err.Lidar, err.meanLidar, err.SD_Lidar, 'LiDAR');
plotOne(2, err.Camera, err.meanCamera, err.SD_Camera, 'Camera');
end

function plotOne(slot, values, mu, sd, name)
subplot(1, 2, slot);
histogram(values);
hold on
xline(mu, '-', sprintf('mean %.4f m', mu), 'LineWidth', 1.5);
xline([mu - sd, mu + sd], '--', 'LineWidth', 1);
hold off
title(sprintf('%s reprojection error (SD %.4f m)', name, sd));
xlabel('Error (m)');
ylabel('Count');
end
