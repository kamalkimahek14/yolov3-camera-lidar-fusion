% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear all, clc;
% Create frames from video and store in directory
v = VideoReader('video.mp4');
numFrames = v.NumberOfFrames;
n=numFrames;
 for i = 1:n
 frames = read(v,i);
 imwrite(frames,['Image' int2str(i), '.jpg']);
 end