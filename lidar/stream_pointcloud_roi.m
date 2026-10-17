% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear; clc;
% Make Pipeline object to manage streaming
pipe = realsense.pipeline();

% Make Colorizer object to prettify depth output
colorizer = realsense.colorizer();

% define point cloud object
pcl_obj = realsense.pointcloud();

% Start streaming with default settings
profile = pipe.start();
align_to = realsense.stream.color;
alignedFs = realsense.align(align_to);

% Get streaming device's name
dev = profile.get_device();
name = dev.get_info(realsense.camera_info.name);

% Get frames. We discard the first couple to allow
% the camera time to settle
for j = 1:5

fs = pipe.wait_for_frames();

end

%create a point cloud player
player1 = pcplayer([-5 5],[-3 3],[0 5],'VerticalAxis','Y','VerticalAxisDir',...
                     'Down','MarkerSize',8);

% load('cam2Lidar_transformation.mat');
% load('Cam_calParams.mat');
% load('net_new.mat');
% networkInputSize = [227 227 3];
% confidenceThreshold = 0.5;
% overlapThreshold = 0.5;
% networkOutputs = ["conv2Detection1"
%                   "conv2Detection2"];
% anchorBoxMasks = {[1,2,3,4,5,6,7,8,9,10,11,12]
%     [13,14,15,16,17,18,19,20,21,22,23,24]
%     };
% classNames = {'Cylinder', 'Box', 'Pyramid', 'Traffic_cone'};
% executionEnvironment = "gpu";
% w = webcam(1);

               
while player1.isOpen() 

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
   
  pCloud = pointCloud(XYZ(1:4:end,1:4:end,:),'Color',RGB(1:4:end,1:4:end,:));
  ROI = [-5, 5, -3, 0.4, 0.3 10];
  indices = findPointsInROI(pCloud,ROI);
  pCloud = select(pCloud,indices);
  
%   s = w.snapshot();
%   Cols = size(s,2);
%   
%   rightimg = s(:,1:Cols/2,:);
%   leftimg = s(:,(Cols/2)+1:end,:);
%   [J1,J2] = rectifyStereoImages(leftimg,rightimg,stereoParams);
%      
%   OrigImSize = size(J1);
%   I1 = preprocessData(J1,networkInputSize);
%   I2 = preprocessData(J2,networkInputSize);
%      
%   XTest1 = dlarray(I1,'SSCB');
%   XTest2 = dlarray(I2,'SSCB');
% 
% % If GPU is available, then convert data to gpuArray.
% if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
%     XTest1 = gpuArray(XTest1);
%     XTest2 = gpuArray(XTest2);
% end
% 
% [bboxes1, scores1, labels1] = yolov3Detect(net, XTest1, networkOutputs,anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);
% [bboxes2, scores2, labels2] = yolov3Detect(net, XTest2, networkOutputs,anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);
% 
% scale = OrigImSize(1:2)./networkInputSize(1:2);
% bboxes1 = bboxresize(bboxes1,scale);
% bboxes2 = bboxresize(bboxes2,scale);
% 
% figure(1)
% imshow(J1)
% showShape('rectangle',bboxes1)
% 
% lidarBbox = bboxCameraToLidar(bboxes1,pCloud,stereoParams.CameraParameters1, ...
%     invert(tform),'ClusterThreshold',0.9,'MaxDetectionRange',[0.4,5]);

% figure(2)
% pcshow(pCloud)
% showShape('Cuboid',lidarBbox)
  
view(player1,pCloud); 
end

pipe.stop();
