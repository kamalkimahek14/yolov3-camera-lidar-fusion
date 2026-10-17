% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear; clc;

load('../sample_data/extrinsic/stereocamCalPerams.mat')
load('../sample_data/extrinsic/gtruth_data.mat');


numFiles = 15;
i = 1;
for j = 1:numFiles
    
    cornersleft = gtruth_Left(:,:,j);
    cornersright = gtruth_Right(:,:,j);
    Camera_Gtruth(i:i+3,:) = triangulate(cornersleft, cornersright, stereoParams)./1000;
    Lidar_Gtruth(i:i+3,:) = gtruth_Lidar(:,:,j);
    i = i + 4;   
end

[cam2lidar,inlierIndex1,status1] = estimateGeometricTransform3D(Camera_Gtruth,Lidar_Gtruth,'rigid');
[lidar2cam,inlierIndex2,status2] = estimateGeometricTransform3D(Lidar_Gtruth,Camera_Gtruth,'rigid');


Lidar_Est = transformPointsForward(cam2lidar,Camera_Gtruth);
Camera_Est = transformPointsForward(lidar2cam,Lidar_Gtruth); 

Reprojection_error.Lidar = sqrt(sum((Lidar_Gtruth).^2')) - sqrt(sum((Lidar_Est).^2'));
Reprojection_error.Camera =sqrt(sum((Camera_Gtruth).^2')) - sqrt(sum((Camera_Est).^2'));
Reprojection_error.meanLidar = mean(Reprojection_error.Lidar);
Reprojection_error.meanCamera = mean(Reprojection_error.Camera);
Reprojection_error.SD_Lidar = std(Reprojection_error.Lidar,1);
Reprojection_error.SD_Camera = std(Reprojection_error.Camera,1);

showReprojectionError(Reprojection_error)


