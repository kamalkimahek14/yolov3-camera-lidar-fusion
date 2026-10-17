% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear; clc;

%%%%%%%%%% Iniialization %%%%%%%%%%%%

%______________Lidar________________%

% Make Pipeline object to manage streaming
pipe = realsense.pipeline();
colorizer = realsense.colorizer();
pcl_obj = realsense.pointcloud();
profile = pipe.start();
align_to = realsense.stream.color;
alignedFs = realsense.align(align_to);
dev = profile.get_device();
name = dev.get_info(realsense.camera_info.name);
% Get frames. We discard the first couple to allow the camera time to settle
for j = 1:5
fs = pipe.wait_for_frames();
end
%create a point cloud player
player_pc = pcplayer([-1 1.2], [0 0.5], [0.4 3], 'VerticalAxis', 'Y', 'VerticalAxisDir', 'Down','MarkerSize', 8);

%______________Camera________________%

load('YOLOv3net.mat');
load('../sample_data/fusion/stereocamCalPerams.mat');
load('../sample_data/fusion/transformations.mat');
networkInputSize = [227 227 3];
confidenceThreshold = 0.5;
overlapThreshold = 0.5;
networkOutputs = ["conv2Detection1"
                  "conv2Detection2"];
anchorBoxMasks = {[1,2,3,4,5,6,7,8,9,10,11,12]
    [13,14,15,16,17,18,19,20,21,22,23,24]
    }; 
classNames = {'Cylinder', 'Box', 'Pyramid', 'Traffic_cone'};
executionEnvironment = "gpu";

w_left = webcam(2);
w_right = webcam(3);

I_left = w_left.snapshot;
I_right = w_right.snapshot;

player_cam = vision.DeployableVideoPlayer();

step(player_cam,I_left)

%______________Fusion________________%

% initialization of estimation state and covariance  
X_est0 = [1.0;0;0; 0.35;0;0; -0.1;0;0];
X_est = [];
X_pred = [];
X_est(1,:) = X_est0;
X_pred(1,:) = X_est0;
P_est0 = blkdiag(1.5*eye(3),0.5*eye(3),0.1*eye(3));
P_est = P_est0;
i = 1;
t = 0;

while player_pc.isOpen() || player_cam.isOpen()
tic;

fs = pipe.wait_for_frames();
%align the depth frames to the color stream
aligned_frames = alignedFs.process(fs);
depth = aligned_frames.get_depth_frame();   
color = fs.get_color_frame();

%get the points cloud based on the aligned depth stream
pnts = pcl_obj.calculate(depth);
pcl_obj.map_to(color);
colordata = color.get_data();
colordatavector = [colordata(1:3:end)',colordata(2:3:end)',colordata(3:3:end)'];
vertices = pnts.get_vertices();
XYZ = pagectranspose(reshape(vertices,[1280 720 3]));
RGB = pagectranspose(reshape(colordatavector,[1280 720 3]));
pCloud_raw = pointCloud(XYZ(:,1:5:end,:),'Color',RGB(:,1:5:end,:));
pCloud_wonoise = pcdenoise(pCloud_raw,'NumNeighbors',1,'Threshold',2);
view(player_pc,pCloud_wonoise)

J_left = w_left.snapshot;
J_right = w_right.snapshot;

% Change image resolution to network input size
OrigImSize = size(J_left);
I_left = preprocessData(J_left,networkInputSize);
I_right = preprocessData(J_right,networkInputSize);

XTest_left = dlarray(I_left,'SSCB');
XTest_right = dlarray(I_right,'SSCB');

% If GPU is available, then convert data in to gpuArray
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    XTest_left = gpuArray(XTest_left);
    XTest_right = gpuArray(XTest_right);
end

% detect object from the image data using YOLOv3 Network
[bboxes_left, scores_left, labels_left] = yolov3Detect(net,XTest_left,networkInputSize,anchorBoxes,anchorBoxMasks,...
                                          confidenceThreshold,overlapThreshold,classNames);
[bboxes_right, scores_right, labels_right] = yolov3Detect(net,XTest_right,networkInputSize,anchorBoxes,anchorBoxMasks,...
                                          confidenceThreshold,overlapThreshold,classNames);
                                      
% scale bboxes that compatible with orignal input image size
scale = OrigImSize(1:2)./networkInputSize(1:2);
bboxes_left = bboxresize(bboxes_left,scale);
bboxes_right = bboxresize(bboxes_right,scale);
scores_left = scores_left*100;
scores_right = scores_right*100; 


dt = toc;
t = t + dt;

% Remove the detection with non similar number of detected objects
if numel(scores_left) == 1 &&  numel(scores_right) == 1
     
         centers_left = bboxes_left(:,1:2) + bboxes_left(:,3:4)./2;
         centers_right = bboxes_right(:,1:2) + bboxes_right(:,3:4)./2;
         
         XYZ_Camera = triangulate(centers_left, centers_right, stereoParams)./1000;
         XYZ_Camera = transformPointsForward(cam2lidar,XYZ_Camera);
         D_Camera = sqrt(sum((XYZ_Camera).^2'));
                 
         % Find corresponding cuboid from PointCloud
          bboxes_Lidar = bboxCameraToLidar(bboxes_left,pCloud_wonoise,stereoParams.CameraParameters1,cam2lidar,...
                           'ClusterThreshold',0.11);

%          [labels,n] = pcsegdist(pCloud_wonoise,2);
%          idx = find(labels == 1);
%          model = pcfitcuboid(pCloud_wonoise,idx);
%          bboxes_Lidar = model.Parameters;

         showShape('cuboid',bboxes_Lidar,'Parent',player_pc.Axes,'Color','green','Opacity',0.5)
                         
         XYZ_Lidar = bboxes_Lidar(:,1:3);
         D_Lidar = sqrt(sum((XYZ_Lidar).^2'));
        
         [X_est,X_pred,P_est] = sensor_fusion(dt,i,X_est0,P_est0,XYZ_Camera,XYZ_Lidar,X_est,X_pred,P_est);
         
                          
      if  labels_left == ['Cylinder'] && labels_right == ['Cylinder']
          
          D_Cyest = sqrt(sum(([X_est(i,1) X_est(i,4) X_est(i,7)]).^2'));
          V_Cyest = sqrt(sum(([X_est(i,2) X_est(i,5) X_est(i,8)]).^2'));
          a_Cyest = sqrt(sum(([X_est(i,3) X_est(i,6) X_est(i,9)]).^2'));
          
          format = 'Ctg: %s Pr: %0.2f D: %0.2f m V: %0.2f a: %0.2f';
          str = sprintf(format,labels_left,(scores_left + scores_left)/2,...
                D_Cyest,V_Cyest,a_Cyest); 
          label = {str};
          J_left = insertObjectAnnotation(J_left, 'rectangle', bboxes_left,...
                   label,'Color','red','TextBoxOpacity',1,'FontSize',25);  
      end
                      
      if  labels_left == ['Box'] && labels_right == ['Box']
          
          D_Boxest = sqrt(sum(([X_est(i,1) X_est(i,4) X_est(i,7)]).^2'));
          V_Boxest = sqrt(sum(([X_est(i,2) X_est(i,5) X_est(i,8)]).^2'));
          a_Boxest = sqrt(sum(([X_est(i,3) X_est(i,6) X_est(i,9)]).^2'));
          
          format = 'Ctg: %s Pr: %0.2f D: %0.2f m V: %0.2f a: %0.2f';
          str = sprintf(format,labels_left,(scores_left + scores_left)/2,...
                        D_Boxest,V_Boxest,a_Boxest); 
          label = {str};
          J_left = insertObjectAnnotation(J_left, 'rectangle', bboxes_left,...
                   label,'Color','blue','TextBoxOpacity',1,'FontSize',25);  
      end
                      
      if  labels_left == ['Pyramid'] && labels_right == ['Pyramid']
  
          D_Pyest = sqrt(sum(([X_est(i,1) X_est(i,4) X_est(i,7)]).^2'));
          V_Pyest = sqrt(sum(([X_est(i,2) X_est(i,5) X_est(i,8)]).^2'));
          a_Pyest = sqrt(sum(([X_est(i,3) X_est(i,6) X_est(i,9)]).^2'));
          
          format = 'Ctg: %s Pr: %0.2f D: %0.2f m V: %0.2f a: %0.2f';
          str = sprintf(format,labels_left,(scores_left + scores_left)/2,...
                        D_Pyest,V_Pyest,a_Pyest); 
          label = {str};
          J_left = insertObjectAnnotation(J_left, 'rectangle', bboxes_left,...
                   label,'Color','white','TextBoxOpacity',1,'FontSize',25);  
      end
                      
      if  labels_left == ['Traffic_cone'] && labels_right == ['Traffic_cone']
      
          D_Tcest = sqrt(sum(([X_est(i,1) X_est(i,4) X_est(i,7)]).^2'));
          V_Tcest = sqrt(sum(([X_est(i,2) X_est(i,5) X_est(i,8)]).^2'));
          a_Tcest = sqrt(sum(([X_est(i,3) X_est(i,6) X_est(i,9)]).^2'));
          
          format = 'Ctg: %s Pr: %0.2f D: %0.2f m V: %0.2f a: %0.2f';
          str = sprintf(format,labels_left,(scores_left + scores_left)/2,...
                        D_Tcest,V_Tcest,a_Tcest); 
          label = {str};
          J_left = insertObjectAnnotation(J_left, 'rectangle', bboxes_left,...
                 label,'Color','green','TextBoxOpacity',1,'FontSize',25);  
      end
      
         result.left{i,1} = bboxes_left;
         result.left{i,2} = scores_left;
         result.left{i,3} = labels_left;
         
         result.right{i,1} = bboxes_right;
         result.right{i,2} = scores_right;
         result.right{i,3} = labels_right;
         
         result.Camera{i,1} = XYZ_Camera;
         result.Camera{i,2} = D_Camera;
         
         result.Lidar{i,1} = XYZ_Lidar;
         result.Lidar{i,2} = D_Lidar;
         
         result.time(i) = t;
         
         i = i+1;
end        
    
step(player_cam,J_left);
end

result.Estimate = X_est;
result.Prediction = X_pred;

pipe.stop();






% preprocessData and yolov3Detect are in detection/yolov3/, rewritten in 2026.
