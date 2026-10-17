% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear;clc;

load('new_net.mat');
load('../sample_data/fusion/stereocamCalPerams.mat');
networkInputSize = [227 227 3];
confidenceThreshold = 0.5;
overlapThreshold = 0.5;
networkOutputs = ["conv2Detection1"
                  "conv2Detection2"];
anchorBoxMasks = {[1,2,3,4,5,6,7,8,9,10,11,12]
    [13,14,15,16,17,18,19,20,21,22,23,24]
    };              

classNames = {'Cylinder', 'Box', 'Pyramid', 'Traffic_cone'};
executionEnvironment = "auto";

w_left = webcam(2);
w_right = webcam(3);

J1 = w_left.snapshot;
J2 = w_right.snapshot;

videoPlayer = vision.DeployableVideoPlayer();

step(videoPlayer,J1)

while videoPlayer.isOpen()
    
     J1 = w_left.snapshot;
     J2 = w_right.snapshot;
     
     OrigImSize = size(J1);
     I1 = preprocessData(J1,networkInputSize);
     I2 = preprocessData(J2,networkInputSize);
     
     XTest1 = dlarray(I1,'SSCB');
     XTest2 = dlarray(I2,'SSCB');

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    XTest1 = gpuArray(XTest1);
    XTest2 = gpuArray(XTest2);
end

[bboxes1, scores1, labels1] = yolov3Detect(net, XTest1, networkOutputs,anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);
[bboxes2, scores2, labels2] = yolov3Detect(net, XTest2, networkOutputs,anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);

scale = OrigImSize(1:2)./networkInputSize(1:2);
bboxes1 = bboxresize(bboxes1,scale);
bboxes2 = bboxresize(bboxes2,scale);

scores1 = scores1*100;
scores2 = scores2*100;

% Display the detections on image.
if ~isempty(scores1) && ~isempty(scores2) && (numel(scores1) ==  numel(scores2))
    
    results1{1,1} = bboxes1;
    results2{1,1} = bboxes2;
    results1{1,2} = scores1;
    results2{1,2} = scores2;
    results1{1,3} = labels1;
    results2{1,3} = labels2;
   
    for ii = 1:size(bboxes2,1)
      
        if results1{1,3}(ii,1) == ['Cylinder'] && results2{1,3}(ii,1) == ['Cylinder']
           Box1 = double(results1{1,1}(ii,:));
           Box2 = double(results2{1,1}(ii,:));
           center1 = Box1(1:2) + Box1(3:4)/2;
           center2 = Box2(1:2) + Box2(3:4)/2;
           D = triangulate(center1, center2, stereoParams);
           D = norm(D)/1000;
           format = 'Ctg: %s Pr: %0.2f D: %0.2f m';
           str = sprintf(format,results1{1,3}(ii,1),(results1{1,2}(ii,1) + results2{1,2}(ii,1))/2,D); 
           label = {str};
           J1 = insertObjectAnnotation(J1, 'rectangle', results1{1,1}(ii,:),...
               label,'Color','red','TextBoxOpacity',1,'FontSize',25);  
        end
        
        if results1{1,3}(ii,1) == ['Box'] && results2{1,3}(ii,1) == ['Box']
           Box1 = double(results1{1,1}(ii,:));
           Box2 = double(results2{1,1}(ii,:));
           center1 = Box1(1:2) + Box1(3:4)/2;
           center2 = Box2(1:2) + Box2(3:4)/2;
           D = triangulate(center1, center2, stereoParams);
           D = norm(D)/1000;
           format = 'Ctg: %s Pr: %0.2f D: %0.2f m';
           str = sprintf(format,results1{1,3}(ii,1),(results1{1,2}(ii,1) + results2{1,2}(ii,1))/2,D); 
           label = {str};
           J1 = insertObjectAnnotation(J1, 'rectangle', results1{1,1}(ii,:),...
               label,'Color','blue','TextBoxOpacity',1,'FontSize',25);      
        end
        
        if results1{1,3}(ii,1) == ['Pyramid'] && results2{1,3}(ii,1) == ['Pyramid']
           Box1 = double(results1{1,1}(ii,:));
           Box2 = double(results2{1,1}(ii,:));
           center1 = Box1(1:2) + Box1(3:4)/2;
           center2 = Box2(1:2) + Box2(3:4)/2;
           D = triangulate(center1, center2, stereoParams);
           D = norm(D)/1000;
           format = 'Ctg: %s Pr: %0.2f D: %0.2f m';
           str = sprintf(format,results1{1,3}(ii,1),(results1{1,2}(ii,1) + results2{1,2}(ii,1))/2,D); 
           label = {str};
           J1 = insertObjectAnnotation(J1, 'rectangle', results1{1,1}(ii,:),...
               label,'Color','white','TextBoxOpacity',1,'FontSize',25);      
        end
        
        if results1{1,3}(ii,1) == ['Traffic_cone'] && results2{1,3}(ii,1) == ['Traffic_cone']
           Box1 = double(results1{1,1}(ii,:));
           Box2 = double(results2{1,1}(ii,:));
           center1 = Box1(1:2) + Box1(3:4)/2;
           center2 = Box2(1:2) + Box2(3:4)/2;
           D = triangulate(center1, center2, stereoParams);
           D = norm(D)/1000;
           format = 'Ctg: %s Pr: %0.2f D: %0.2f m';
           str = sprintf(format,results1{1,3}(ii,1),(results1{1,2}(ii,1) + results2{1,2}(ii,1))/2,D); 
           label = {str};
           J1 = insertObjectAnnotation(J1, 'rectangle', results1{1,1}(ii,:),...
               label,'Color','green','TextBoxOpacity',1,'FontSize',25);
        end

    end

end

 
step(videoPlayer,J1);
end
release(videoPlayer);


% preprocessData and yolov3Detect are in detection/yolov3/, rewritten in 2026.