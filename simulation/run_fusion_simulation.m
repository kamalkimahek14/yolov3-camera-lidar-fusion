% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear; clc;
%%%%%% Driving Scene %%%%%%%%
[allData, scenario, sensors] = DrivingScene;
% camera = sensors{1,1};
% lidar = sensors{1,2};
allData = struct2table(allData);

% Initialize video file
% fileName = 'unsupervised_case1';
% if ~isempty(fileName)
% 	v = VideoWriter([fileName,'.avi']);
%     v.FrameRate = 5;
% 	open(v);
% end

%%%%%% Sensor Fusion Parameter initialization %%%%%%%%
dt = 0.1;
F = [ 1 dt 0.5*dt^2 zeros(1,6);
      0 1 dt zeros(1,6);
      0 0 1 zeros(1,6);
      zeros(1,3) 1 dt 0.5*dt^2 zeros(1,3);
      zeros(1,4) 1 dt zeros(1,3);
      zeros(1,5) 1 zeros(1,3);
      zeros(1,6) 1 dt 0.5*dt^2;
      zeros(1,7) 1 dt;
      zeros(1,8) 1];
  
H = [1 zeros(1,8);
     zeros(1,3) 1 zeros(1,5) ;
     zeros(1,6) 1 0 0]; 
 
load('../sample_data/simulation/sensor_covariance.mat');

% initialization of estimation state and covariance  
X_est0 = [5;0;0;20;0;0;0.0001;0;0];
P_est0 = blkdiag(0.835*eye(3),0.2616*eye(3),0.001*eye(3));

for i  = 1:size(allData,1)  
    
% inertial frame    
Car_Gtruth_Pos = allData.ActorPoses{i,1}(1).Position;
Obj_Gtruth_Pos = allData.ActorPoses{i,1}(2).Position;
Car_Gtruth_Vel = allData.ActorPoses{i,1}(1).Velocity;
Obj_Gtruth_Vel = allData.ActorPoses{i,1}(2).Velocity;

% body frame
R = rotz(allData.ActorPoses{i,1}(1).Yaw);
Gtruth.PositionB(i,1:3) = (Obj_Gtruth_Pos - Car_Gtruth_Pos)*R;
Gtruth.PositionI(i,1:3) = Obj_Gtruth_Pos;

time(i) = allData.Time(i);
ptCloud = allData.PointClouds(i);
[labels,numClusters] = pcsegdist(ptCloud,2);
idx = find(labels == 1);
model = pcfitcuboid(ptCloud,idx);
X_Lidar(i,1:3) = model.Center - [0 0 model.Center(3)];

 
if ~isempty(allData.ObjectDetections{i,1})
    X_camera(i,1:3) = allData.ObjectDetections{i,1}{1,1}.Measurement(1:3);
        
       if i == 1

        %%%% Lidar %%%%%%    
            %prediction
            X_pred(i,:) = F * X_est0;
            P_pred = F * P_est0 * F';

            %Kalman gain
            K = P_pred * H' /(H * P_pred * H' + R_Lidar);

            %Correction
            X_est(i,:) = X_pred(i,:)' + K * ((X_Lidar(i,:)' - H*X_pred(i,:)'));
            P_est = (eye(9) - K * H) * P_pred;

        %%%% Camera %%%%%%    
            %prediction
            X_pred(i,:) = F * X_est(i,:)';
            P_pred = F * P_est * F';

            %Kalman gain
            K = P_pred * H' /(H * P_pred * H' + R_Camera);

            %Correction
            X_est(i,:) = X_pred(i,:)' + K * ((X_camera(i,:)' - H*X_pred(i,:)'));
            P_est = (eye(9) - K * H) * P_pred;
            
       else
           
        %%%% Lidar %%%%%%    
            %prediction
            X_pred(i,:) = F * X_est(i-1,:)';
            P_pred = F * P_est * F';

            %Kalman gain
            K = P_pred * H' /(H * P_pred * H' + R_Lidar);

            %Correction
            X_est(i,:) = X_pred(i,:)' + K * ((X_Lidar(i,:)' - H*X_pred(i,:)'));
            P_est = (eye(9) - K * H) * P_pred;

        %%%% Camera %%%%%%    
            %prediction
            X_pred(i,:) = F * X_est(i,:)';
            P_pred = F * P_est * F';

            %Kalman gain
            K = P_pred * H' /(H * P_pred * H' + R_Camera);

            %Correction
            X_est(i,:) = X_pred(i,:)' + K * ((X_camera(i,:)' - H*X_pred(i,:)'));
            P_est = (eye(9) - K * H) * P_pred;
           
       end
       
else 
    
    %%%% Lidar %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i-1,:)';
        P_pred = F * P_est * F';

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Lidar);

        %Correction
        X_est(i,:) = X_pred(i,:)' + K * ((X_Lidar(i,:)' - H*X_pred(i,:)'));
        P_est = (eye(9) - K * H) * P_pred;

    %%%% Camera %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i,:)';
        P_pred = F * P_est * F';

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Camera);

        %Correction
        X_est(i,:) = X_pred(i,:);
        P_est = (eye(9) - K * H) * P_pred;
       
       
end

X_estI(i,:) = ([X_est(i,1) X_est(i,4) X_est(i,7)]*R' + Car_Gtruth_Pos); 
figure(1);
plot(X_estI(1:i,2),X_estI(1:i,1),'m',Gtruth.PositionI(1:i,2),Gtruth.PositionI(1:i,1),'r');
set(gca, 'XDir','reverse')
legend('Estimated','Ground Truth','Location','northwest')
title('Inertial Frame')
xlabel('Y (Meter)')
ylabel('X (Meter)')
hold on; grid on

% 	if ~isempty(fileName)
% 		M = getframe(gcf);
% 		writeVideo(v,M);
%     end
%     
end

hold off
% % Close video file
% if ~isempty(fileName)
% 	close(v)
% end

%Position
figure(2);
%X
subplot(311);
plot(time,X_camera(:,1),'g',time,X_Lidar(:,1),'--g',time,X_pred(:,1),'b')
hold on;
plot(time,X_est(:,1),'m',"LineWidth",2)
hold on;
plot(time,Gtruth.PositionB(:,1),'r',"LineWidth",2)
xlabel('t (sec)')
ylabel('X Position (Meter)')
legend('Camera', 'Lidar', 'Prediction', 'Estimated','Gtruth')
hold off;
%Y
subplot(312);
plot(time,X_camera(:,2),'g',time,X_Lidar(:,2),'--g',time,X_pred(:,4),'b')
hold on;
plot(time,X_est(:,4),'m',"LineWidth",2)
hold on;
plot(time,Gtruth.PositionB(:,2),'r',"LineWidth",2)
xlabel('t (sec)')
ylabel('Y Position (Meter)')
legend('Camera', 'Lidar', 'Prediction', 'Estimated','Gtruth')
hold off;
%Z
subplot(313);
plot(time,X_camera(:,3),'g',time,X_Lidar(:,3),'--g',time,X_pred(:,7),'b')
hold on;
plot(time,X_est(:,7),'m',"LineWidth",2)
hold on;
plot(time,Gtruth.PositionB(:,3),'r',"LineWidth",2)
xlabel('t (sec)')
ylabel('Z Position(Meter)')
legend('Camera', 'Lidar', 'Prediction', 'Estimated','Gtruth')
hold off;

%Velocity
figure(3);
%X
subplot(311);
plot(time,X_est(:,2),'m',"LineWidth",2)
hold on;
plot(time,X_pred(:,2),'b')
xlabel('t (sec)')
ylabel('X Velocity (Meter/sec)')
legend('Estimated','Prediction')
hold off;
%Y
subplot(312);
plot(time,X_pred(:,5),'b')
hold on;
plot(time,X_est(:,5),'m',"LineWidth",2)
xlabel('t (sec)')
ylabel('Y Velocity (Meter/sec)')
legend( 'Prediction','Estimated')
hold off;
%Z
subplot(313);
plot(time,X_pred(:,8),'b')
hold on;
plot(time,X_est(:,8),'m',"LineWidth",2)
xlabel('t (sec)')
ylabel('Z Velocity (Meter/sec)')
legend( 'Prediction', 'Estimated')
hold off;
   
%Acceleration
figure(4);
%X
subplot(311);
plot(time,X_est(:,3),'m',"LineWidth",2)
hold on;
plot(time,X_pred(:,3),'b')
xlabel('t (sec)')
ylabel('X Acceleration (Meter/sec^2)')
legend( 'Estimated','Prediction')
hold off;
%Y
subplot(312);
plot(time,X_pred(:,6),'b')
hold on;
plot(time,X_est(:,6),'m',"LineWidth",2)
xlabel('t (sec)')
ylabel('Y Acceleration (Meter/sec^2)')
legend( 'Prediction', 'Estimated')
hold off;
%Z
subplot(313);
plot(time,X_pred(:,9),'b')
hold on;
plot(time,X_est(:,9),'m',"LineWidth",2)
xlabel('t (sec)')
ylabel('Z Acceleration (Meter/sec^2)')
legend( 'Prediction', 'Estimated')
hold off;

%%%Residue Error
Rerror = Gtruth.PositionB - [X_est(:,1) X_est(:,4) X_est(:,7)];
Error_camera = Gtruth.PositionB - X_camera;
Error_Lidar = Gtruth.PositionB - X_Lidar;
Mean = mean(Rerror);

figure(5);
subplot(131);
A1 = histogram(Rerror(:,1),50);
title('X Position Error');
legend(A1,strcat('Mean Error: ', num2str(Mean(1)),' m'),'Location','southeast');

subplot(132);
A2 = histogram(Rerror(:,2),50);
title('Y Position Error');
legend(A2,strcat('Mean Error: ', num2str(Mean(2)),' m'),'Location','southeast');

subplot(133);
A3 = histogram(Rerror(:,3),50);
title('Z Position Error');
legend(A3,strcat('Mean Error: ', num2str(Mean(3)),' m'),'Location','southeast');

figure(6);
subplot(311)
plot(time,Error_camera(:,1),'g',time,Error_Lidar(:,1),'--g')
hold on
plot(time,Rerror(:,1),'m',"LineWidth",2)
xlabel('t (sec)')
ylabel('X Error (m)')
legend('Camera','Lidar','Estimated')
hold off

subplot(312)
plot(time,Error_camera(:,2),'g',time,Error_Lidar(:,2),'--g')
hold on
plot(time,Rerror(:,2),'m',"LineWidth",2)
xlabel('t (sec)')
ylabel('Y Error (m)')
legend('Camera','Lidar','Estimated')
hold off

subplot(313)
plot(time,Error_camera(:,3),'g',time,Error_Lidar(:,3),'--g')
hold on
plot(time,Rerror(:,3),'m',"LineWidth",2)
xlabel('t (sec)')
ylabel('Z Error (m)')
legend('Camera','Lidar','Estimated')
hold off