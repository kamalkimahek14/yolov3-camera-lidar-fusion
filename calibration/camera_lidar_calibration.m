% Original MS thesis code (2021), Kamal Mehta. See README.md.
%% Intrinsic Calibration
load('../sample_data/calibration/Intrinsic_params.mat'); % ground truth and sensor measurements

Overall_cam_error = sqrt(sum((Camera_Gtruth).^2')) - sqrt(sum((Camera_measurement).^2'));
Overall_lidar_error = sqrt(sum((Lidar_Gtruth).^2')) - sqrt(sum((Lidar_measurement).^2'));

figure(1)
subplot(221),histfit(Camera_error(:,1),100);title('Camera X Error (cm)')
legend(strcat('X Mean Error: ', num2str(mean(Camera_error(:,1))),' in cm'),'Location','southeast')

subplot(222),histfit(Camera_error(:,2),100);title('Camera Y Error (cm)')
legend(strcat('Y Mean Error: ', num2str(mean(Camera_error(:,2))),' in cm'),'Location','southeast')

subplot(223),histfit(Camera_error(:,3),100);title('Camera Z Error (cm)')
legend(strcat('Z Mean Error: ', num2str(mean(Camera_error(:,3))),' in cm'),'Location','southeast')

subplot(224),histfit(Overall_cam_error,100);title('Camera Overall Error (cm)')
legend(strcat('Overall Mean Error: ', num2str(mean(Overall_cam_error)),' in cm'),'Location','southeast')

figure(2)
subplot(221),histfit(Lidar_error(:,1),100);title('Lidar X Error (cm)')
legend(strcat('X Mean Error: ', num2str(mean(Lidar_error(:,1))),' in cm'),'Location','southeast')

subplot(222),histfit(Lidar_error(:,2),100);title('Lidar Y Error (cm)')
legend(strcat('Y Mean Error: ', num2str(mean(Lidar_error(:,2))),' in cm'),'Location','southeast')

subplot(223),histfit(Lidar_error(:,3),100);title('Lidar Z Error (cm)')
legend(strcat('Z Mean Error: ', num2str(mean(Lidar_error(:,3))),' in cm'),'Location','southeast')

subplot(224),histfit(Overall_lidar_error,100);title('Lidar Overall Error (cm)')
legend(strcat('Overall Mean Error: ', num2str(mean(Overall_lidar_error)),' in cm'),'Location','southeast')

%% Extrinsic Calibration 

R = [0 -1 0; 0 0 -1; 1 0 0];
[cam2lidar,inlierIndex1,status1] = estimateGeometricTransform3D((R*Camera_measurement')'./100,(R*Lidar_measurement')'./100,'rigid');
[lidar2cam,inlierIndex2,status2] = estimateGeometricTransform3D((R*Lidar_measurement')'./100,(R*Camera_measurement')'./100,'rigid');


Lidar_Est = transformPointsForward(cam2lidar,(R*Camera_measurement')');
Camera_Est = transformPointsForward(lidar2cam,(R*Lidar_measurement')'); 

Reprojection_error.Lidar = sqrt(sum((R*Lidar_measurement')'.^2')) - sqrt(sum((Lidar_Est).^2'));
Reprojection_error.Camera =sqrt(sum((R*Camera_measurement')'.^2')) - sqrt(sum((Camera_Est).^2'));
Reprojection_error.meanLidar = mean(Reprojection_error.Lidar);
Reprojection_error.meanCamera = mean(Reprojection_error.Camera);
Reprojection_error.SD_Lidar = std(Reprojection_error.Lidar,1);
Reprojection_error.SD_Camera = std(Reprojection_error.Camera,1);

helperShowError(Reprojection_error)