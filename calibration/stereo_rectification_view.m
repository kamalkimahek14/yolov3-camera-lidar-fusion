% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear all; close all; clc

load('../sample_data/calibration/Cal_params.mat')
w = webcam(2);
videoPlayer = vision.DeployableVideoPlayer();
s = w.snapshot();
step(videoPlayer,uint8(zeros(1043,1499,3)))
% player3D = pcplayer([-3, 3], [-3, 3], [0, 8], 'VerticalAxis', 'Y', ...
%                        'VerticalAxisDir', 'down');

while videoPlayer.isOpen()
    
    s = w.snapshot();
    Cols = size(s,2);
    rightimg = s(:,1:Cols/2,:);
    leftimg = s(:,(Cols/2)+1:end,:); 
    [J1,J2] = rectifyStereoImages(rightimg,leftimg,stereoParams);
    disparityMap = disparitySGM(rgb2gray(J1),rgb2gray(J2));
%     points3D = reconstructScene(disparityMap, stereoParams);
%     points3D = points3D ./ 1000;
%     ptCloud = pointCloud(points3D, 'Color', J1);
    
%     view(player3D, ptCloud);
    step(videoPlayer,J1);
end
release(videoPlayer);