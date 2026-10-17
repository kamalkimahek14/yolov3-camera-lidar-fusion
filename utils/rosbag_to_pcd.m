% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear; clc;
for i = 1:15
bag = rosbag(sprintf('%i.bag',i));
msg = select(bag,'Topic','camera/depth_registered/points');
Pcloud = readMessages(msg);3
XYZ = readXYZ(Pcloud{3});
RGB = readRGB(Pcloud{3});,

R = [cosd(-90) -sind(-90) 0; sind(-90) cosd(-90) 0; 0 0 1]*eye(3)*[1 0 0; 0 cosd(-90) -sind(-90); 0 sind(-90) cosd(-90)];
xyz = R *(XYZ)';

xyz = pagectranspose(reshape(xyz',[640 480 3]));
RGB = pagectranspose(reshape(RGB,[640 480 3]));


pcl = pointCloud(xyz(1:2:end,1:2:end,:),'Color', RGB(1:2:end,1:2:end,:));
% imwrite(RGB,sprintf('%i.jpg',i));
pcwrite(pcl,sprintf('%i.pcd',i),'Encoding','ascii');
end

% figure(1)
% pcshow(pcl)
% xlabel('X');
% ylabel('Y');
% zlabel('Z');