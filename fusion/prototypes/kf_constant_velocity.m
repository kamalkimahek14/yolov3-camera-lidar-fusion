% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear; clc

%--------Constant velocity-----------%
dt = 0.1;
F = [ 1 dt zeros(1,4);
      0 1 zeros(1,4);
      0 0 1 dt 0 0;
      0 0 0 1 0 0;
      zeros(1,4) 1 dt;
      zeros(1,4) 0 1];
  
H = [1 zeros(1,5);
     zeros(1,2) 1 zeros(1,3) ;
     zeros(1,4) 1 0];  
  
R = [0.000059 0 0;
     0 0.000055 0;
     0 0 0.000048];  
 
Q = [0.000029 zeros(1,5);
     0 0.000029 zeros(1,4);
     zeros(1,2) 0.000025 zeros(1,3);
     zeros(1,3) 0.000025 zeros(1,2);
     zeros(1,4) 0.000038 0;
     zeros(1,5) 0.000038]; 
 

% initialization of estimation state and covariance  
X_est0 = [0.02;0;0.35;0;1.0;0];
P_est0 = Q;
%sensor measurements
u = 0.005;
v = 0.0001;
w = 0.01;
X_sensor = [0.018:u*dt:0.118;
            0.39:v*dt:0.392;
            1.1:w*dt:1.3]';


for i = 1:size(X_sensor,1)
    
    if i == 1
       %prediction
        X_pred(i,:) = F * X_est0;
        P_pred(:,:,i) = F * P_est0 * F' + Q;
                
        %Kalman gain
        K(:,:,i) = P_pred(:,:,i) * H' /(H * P_pred(:,:,i) * H' + R);
    
       %Correction
        X_est(i,:) = X_pred(i,:)' + K(:,:,i) * (X_sensor(i,:)' - H*X_pred(i,:)');
        P_est(:,:,i) = (eye(6) - K(:,:,i) * H) * P_pred(:,:,i);
        
    else
        
       %prediction
        X_pred(i,:) = F * X_est(i-1,:)';
        P_pred(:,:,i) = F * P_est(:,:,i-1) * F' + Q;  
                
        %Kalman gain
        K(:,:,i) = P_pred(:,:,i) * H' /(H * P_pred(:,:,i) * H' + R);
    
       %Correction
        X_est(i,:) = X_pred(i,:)' + K(:,:,i) * (X_sensor(i,:)' - H*X_pred(i,:)');
        P_est(:,:,i) = (eye(6) - K(:,:,i) * H) * P_pred(:,:,i);
    end
end

figure;
subplot(121);
title('Position');
plot(0:dt:200*dt,sqrt(sum((X_sensor).^2')),'g','LineWidth',2)
hold on;
plot(0:dt:200*dt,sqrt(sum(([X_pred(:,1) X_pred(:,3) X_pred(:,5)]).^2')),'--b','LineWidth',2)
hold on;
plot(0:dt:200*dt,sqrt(sum(([X_est(:,1) X_est(:,3) X_est(:,5)]).^2')),'Xm','LineWidth',2)
xlabel('t (sec)')
ylabel('Distance (Meter)')
legend('Sensor', 'Prediction', 'Estimated')
hold off;

subplot(122);
title('Velocity');
plot(0:dt:200*dt,sqrt(sum(([X_pred(:,2) X_pred(:,4) X_pred(:,6)]).^2')),'--b','LineWidth',2)
hold on;
plot(0:dt:200*dt,sqrt(sum(([X_est(:,2) X_est(:,4) X_est(:,6)]).^2')),'Xm','LineWidth',2)
xlabel('t (sec)')
ylabel('Velocity (Meter/sec)')
legend('Prediction', 'Estimated')
hold off;
  