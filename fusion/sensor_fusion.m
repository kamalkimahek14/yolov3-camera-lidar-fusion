% Original MS thesis code (2021), Kamal Mehta. See README.md.
function [X_est,X_pred,P_est] = sensor_fusion(dt,i,X_est0,P_est0,X_camera,X_lidar,X_est,X_pred,P_est)

% Constant acceleration
f = [1 dt 0.5*dt^2;
      0 1 dt;
      0 0 1];
F = blkdiag(f,f,f);
  
H = [1 zeros(1,8);
     zeros(1,3) 1 zeros(1,5) ;
     zeros(1,6) 1 0 0];  
  
R_lidar = [5.84e-05 0 0;
           0 2.73e-04 0;
           0 0 3.917e-06];  
 
R_Camera = [2.59e-05 0 0;
            0 9.76e-06 0;
            0 0 4.46e-04]; 
 
Q = blkdiag(0.3*eye(3),0.2*eye(3),0.1*eye(3));

if ~isempty(X_lidar) 
    
if i == 1
   %%%% Lidar %%%%%% 
       %prediction 
        X_pred(i,:) = F * X_est0;
        P_pred = F * P_est0* F' + Q;
                
        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_lidar);
    
       %Correction
        X_est(i,:) = X_pred(i,:)' + K * (X_lidar' - H*X_pred(i,:)');
        P_est = (eye(9) - K * H) * P_pred;
        
   %%%% Camera %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Camera);

        %Correction
        X_est(i,:) = X_pred(i,:)' + K * ((X_camera' - H*X_pred(i,:)'));
        P_est = (eye(9) - K * H) * P_pred;           
else    
    %%%% Lidar %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i-1,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_lidar);

        %Correction
        X_est(i,:) = X_pred(i,:)' + K * ((X_lidar' - H*X_pred(i,:)'));
        P_est = (eye(9) - K * H) * P_pred;

    %%%% Camera %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Camera);

        %Correction
        X_est(i,:) = X_pred(i,:)' + K * ((X_camera' - H*X_pred(i,:)'));
        P_est = (eye(9) - K * H) * P_pred; 
end

elseif isempty(X_lidar) && isempty(X_camera) 
    
    if i == 1
   %%%% Lidar %%%%%% 
       %prediction 
        X_pred(i,:) = F * X_est0;
        P_pred = F * P_est0* F' + Q;
                
        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_lidar);
    
       %Correction
        X_est(i,:) = X_pred(i,:);
        P_est = (eye(9) - K * H) * P_pred;
        
   %%%% Camera %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Camera);

        %Correction
        X_est(i,:) = X_pred(i,:);
        P_est = (eye(9) - K * H) * P_pred;           
    else  
    
        %%%% Lidar %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i-1,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_lidar);

        %Correction
        X_est(i,:) = X_pred(i,:);
        P_est = (eye(9) - K * H) * P_pred;

    %%%% Camera %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Camera);

        %Correction
        X_est(i,:) = X_pred(i,:);
        P_est = (eye(9) - K * H) * P_pred; 
    end

else 
    if i == 1
   %%%% Lidar %%%%%% 
       %prediction 
        X_pred(i,:) = F * X_est0;
        P_pred = F * P_est0* F' + Q;
                
        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_lidar);
    
       %Correction
        X_est(i,:) = X_pred(i,:)';
        P_est = (eye(9) - K * H) * P_pred;
        
   %%%% Camera %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Camera);

        %Correction
        X_est(i,:) = X_pred(i,:)' + K * ((X_camera' - H*X_pred(i,:)'));
        P_est = (eye(9) - K * H) * P_pred;           
   else  
    %%%% Lidar %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i-1,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_lidar);

        %Correction
        X_est(i,:) = X_pred(i,:);
        P_est = (eye(9) - K * H) * P_pred;

    %%%% Camera %%%%%%    
        %prediction
        X_pred(i,:) = F * X_est(i,:)';
        P_pred = F * P_est * F' + Q;

        %Kalman gain
        K = P_pred * H' /(H * P_pred * H' + R_Camera);

        %Correction
        X_est(i,:) = X_pred(i,:)' + K * ((X_camera' - H*X_pred(i,:)'));
        P_est = (eye(9) - K * H) * P_pred;
    end
end
end
