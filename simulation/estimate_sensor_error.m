% Original MS thesis code (2021), Kamal Mehta. See README.md.

j = 1;
for i  = 1:size(allData,1)

% inertial frame    
Car_Gtruth_Pos = allData.ActorPoses{i,1}(1).Position;
Car_Gtruth_Vel = allData.ActorPoses{i,1}(1).Velocity;
Obj_Gtruth_Pos = allData.ActorPoses{i,1}(2).Position;
Obj_Gtruth_Vel = allData.ActorPoses{i,1}(2).Velocity;

% body frame
R = rotz(allData.ActorPoses{i,1}(1).Yaw);
Gtruth.Position(i,1:3) = (Obj_Gtruth_Pos - Car_Gtruth_Pos)*R;
Gtruth.Velocity(i,1:3) = (Obj_Gtruth_Vel - Car_Gtruth_Vel)*R;

time(i) = allData.Time(i);
ptCloud = allData.PointClouds(i);
[labels,numClusters] = pcsegdist(ptCloud,2);
idx = find(labels == 1);
model = pcfitcuboid(ptCloud,idx);
X_Lidar(i,1:3) = model.Center - [0 0 model.Center(3)];
Error_Lidar(i,1:3) = Gtruth.Position(i,1:3) - X_Lidar(i,1:3);

if ~isempty(allData.ObjectDetections{i,1})
   X_camera(j,1:3) = allData.ObjectDetections{i,1}{1,1}.Measurement(1:3);
   Error_cam(j,1:3) = Gtruth.Position(i,1:3) - X_camera(j,1:3);
   time_cam(j) = time(i);
   j = j+1;
end

end
%%
%Camera
mean_errorcam = mean(Error_cam);
SD_cam = std(Error_cam);
R_Camera = SD_cam'*SD_cam;

%Lidar 
mean_errorlidar = mean(Error_Lidar);
SD_lidar = std(Error_Lidar);
R_Lidar = SD_lidar'*SD_lidar;

figure;
subplot(231),hist(Error_cam(:,1),50);title('X error Camera');
legend(strcat('X Mean Error: ', num2str(mean_errorcam(1)),' in m'),'Location','southeast')

subplot(232),hist(Error_cam(:,2),50);title('Y error Camera');
legend(strcat('Y Mean Error: ', num2str(mean_errorcam(2)),' in m'),'Location','southeast')

subplot(233),hist(Error_cam(:,3),50);title('Z error Camera');
legend(strcat('Z Mean Error: ', num2str(mean_errorcam(3)),' in m'),'Location','southeast')

subplot(234),hist(Error_Lidar(:,1),50);title('X error Lidar');
legend(strcat('X Mean Error: ', num2str(mean_errorlidar(1)),' in m'),'Location','southeast')

subplot(235),hist(Error_Lidar(:,2),50);title('Y error Lidar');
legend(strcat('Y Mean Error: ', num2str(mean_errorlidar(2)),' in m'),'Location','southeast')

subplot(236),hist(Error_Lidar(:,3),50);title('Z error Lidar');
legend(strcat('Z Mean Error: ', num2str(mean_errorlidar(3)),' in m'),'Location','southeast')

%% Error time plot

figure(1);
subplot(211)
plot(time_cam,Error_cam(:,1),'g',time_cam,Error_cam(:,2),'b')
legend('X Error','Y Error')
title('Camera Measurement Error Plot')
xlabel('time (sec)')
ylabel('Error (m)')

subplot(212)
plot(time,Error_Lidar(:,1),'g',time,Error_Lidar(:,2),'b')
legend('X Error','Y Error');
title('Lidar Measurement Error Plot')
xlabel('time (sec)')
ylabel('Error (m)')
