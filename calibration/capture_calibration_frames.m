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
        

% while player1.isOpen() 

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
   
  pCloud = pointCloud(XYZ,'Color',RGB);
  
w_left = webcam(2);
w_right = webcam(3);
I_left = w_left.snapshot();
I_right = w_right.snapshot(); 
  
  i = 1;
  
  imwrite(I_left,sprintf('Left%i.jpg',i))
  imwrite(I_right,sprintf('Right%i.jpg',i))
  pcwrite(pcl,sprintf('%i.pcd',i),'Encoding','ascii');
  
% view(player1,pCloud); 
% end

pipe.stop();