% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear all; clc
% load the dataset
% DS = load('gTruth_image.mat');
% vd = VideoReader('video.mp4');
% gTruth = createdatatable(DS);
% gTruth = float2int(gTruth);

dfs = load('gTruth_image.mat');
vehicleDataset = dfs.gTruth_image;
  
% Split dataset for training and testing l
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.98 * length(shuffledIndices));
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx),:);
testDataTbl = vehicleDataset(shuffledIndices(idx+1:end),:);

% Create image datastore
imdsTrain = imageDatastore(trainingDataTbl.Image_FileName);
imdsTest = imageDatastore(testDataTbl.Image_FileName);

% create ground truth datastore
bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

% Set the minibatch data size
miniBatchSize = 10;
imdsTrain.ReadSize = miniBatchSize;
bldsTrain.ReadSize = miniBatchSize;

% combine image and box label datastore
trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest,bldsTest);

% Data augmentation to increse the datastorage and accuracy
augmentedTrainingData = transform(trainingData,@augmentData);

% Preprocess the input training data
networkInputSize = [227 227 3];
preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocessData(data, networkInputSize));
data = read(preprocessedTrainingData);

% Estimate anchorbox from training data
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,networkInputSize));
numAnchors = 24;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

% Anchorbox mask
area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
[~, idx] = sort(area, 'descend');
anchorBoxes = anchorBoxes(idx, :);
anchorBoxMasks = {[1,2,3,4,5,6,7,8,9,10,11,12]
    [13,14,15,16,17,18,19,20,21,22,23,24]
    };

% network for feature extraction
baseNetwork = squeezenet;
lgraph = squeezenetFeatureExtractor(baseNetwork, networkInputSize);

% define classes and their numbers, add the anchorboxmask into network
classNames = vehicleDataset.Properties.VariableNames(2:end);
numClasses = size(classNames, 2);
numPredictorsPerAnchor = 5 + numClasses;
lgraph = addFirstDetectionHead(lgraph, anchorBoxMasks{1}, numPredictorsPerAnchor);
lgraph = addSecondDetectionHead(lgraph, anchorBoxMasks{2}, numPredictorsPerAnchor);

% merge the layers between two detection head
lgraph = connectLayers(lgraph, 'fire9-concat', 'conv1Detection1');
lgraph = connectLayers(lgraph,'relu1Detection1','upsample1Detection2');
lgraph = connectLayers(lgraph,'fire5-concat','depthConcat1Detection2/in2');

% Define the output later at each detection head
networkOutputs = ["conv2Detection1"
    "conv2Detection2"
    ];

% specify the training options
numIterations = 11000;
learningRate = 0.0001;
warmupPeriod = 500;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];
dotraining = 1;
executionEnvironment = "gpu";

% Trainingif dotraining == true
    % Convert layer graph to dlnetwork.
    net = dlnetwork(lgraph);
    
    % Create subplots for the learning rate and mini-batch loss.
    fig = figure;
    [lossPlotter, learningRatePlotter] = configureTrainingProgressPlotter(fig);
    
    % Custom training loop.
    for iteration = 1:numIterations
        
        % Reset datastore.
        if ~hasdata(preprocessedTrainingData)
            reset(preprocessedTrainingData);
        end
        
        % Read batch of data and create batch of images and
        % ground truths.
        data = read(preprocessedTrainingData);
        [XTrain,YTrain] = createBatchData(data, classNames);
        
        % Convert mini-batch of data to dlarray.
        XTrain = dlarray(single(XTrain),'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XTrain = gpuArray(XTrain);
        end
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients,loss,state] = dlfeval(@modelGradients, net, XTrain, YTrain, anchorBoxes, anchorBoxMasks, penaltyThreshold, networkOutputs);
        
        % Apply L2 regularization.
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, net.Learnables);
        
        % Determine the current learning rate value.
        currentLR = piecewiseLearningRateWithWarmup(iteration, learningRate, warmupPeriod, numIterations);
        
        % Update the network learnable parameters using the SGDM optimizer.
        [net, velocity] = sgdmupdate(net, gradients, velocity, currentLR);
        
        % Update the state parameters of dlnetwork.
        net.State = state;
        
        % Display progress.
        displayLossInfo(iteration, currentLR, loss);  
        
        % Update training plot with new points.
        addpoints(lossPlotter, iteration, double(gather(extractdata(loss.totalLoss))));
        addpoints(learningRatePlotter, iteration, currentLR);
        drawnow  
    end

%% Evaluate Model

confidenceThreshold = 0.5;
overlapThreshold = 0.5;

% Create the test datastore.
preprocessedTestData = transform(testData,@(data)preprocessData(data,networkInputSize));

% Create a table to hold the bounding boxes, scores, and labels returned by
% the detector. 
numImages = size(testDataTbl,1);
results = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','cell'},...
    'VariableNames',{'Boxes','Scores','Labels'});

Gtruth = table('Size',[numImages 3],...
    'VariableTypes',{'cell','cell','double'},...
    'VariableNames',{'Boxes','Labels','time'});

Modtestresult = table('Size',[size(Gtruth,1) 7],...
    'VariableTypes',{'cell','cell','cell','cell','cell','cell','cell'},...
    'VariableNames',{'TestImage','GLabels','GBoxes','RLabels','RBoxes','Rscore','overlapRatio'});

% Run detector on each image in the test set and collect results.
for i = 1:numImages
    i
    % Read the datastore and get the image.
    data = read(preprocessedTestData);
    I = data{1};
    tic
    % Convert to dlarray. If GPU is available, then convert data to gpuArray.
    XTest = dlarray(I,'SSCB');
    if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
        XTest = gpuArray(XTest);
    end
    
    % Run the detector.
    [bboxes, scores, labels] = yolov3Detect(net, XTest, networkOutputs, anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);
    t = toc;
    % Collect the results.
    results.Boxes{i} = bboxes;
    results.Scores{i} = scores;
    results.Labels{i} = labels;
    Gtruth.time(i) = t;
    Gtruth.Boxes{i} = data{1,2};
    Gtruth.Labels{i} = data{1,3};
    Modtestresult.TestImage{i} = I;
    
    if ~isempty(results.Boxes{i})
       for j = 1:size(Gtruth.Labels{i},1)
           GLabel = Gtruth.Labels{i}(j); 
           for ii = 1:size(results.Labels{i},1)
               RLabel = results.Labels{i}(ii);
               if GLabel == RLabel
                  Modtestresult.GLabels{i}(j) = GLabel;
                  Modtestresult.GBoxes{i}(j,:) = Gtruth.Boxes{i}(j,:);
                  Modtestresult.RLabels{i}(j) = results.Labels{i}(ii);
                  Modtestresult.RBoxes{i}(j,:) = results.Boxes{i}(ii,:);
                  Modtestresult.Rscore{i}(j) = results.Scores{i}(ii);
                  Modtestresult.overlapRatio{i}(j) = bboxOverlapRatio(Modtestresult.GBoxes{i}(j,:),Modtestresult.RBoxes{i}(j,:));
               else
                   Modtestresult.GLabels{i}(j) = GLabel;
                   Modtestresult.GBoxes{i}(j,:) = Gtruth.Boxes{i}(j,:);
               end    
           end
       end
    else
        Modtestresult.GLabels{i} = Gtruth.Labels{i};
        Modtestresult.GBoxes{i} = Gtruth.Boxes{i};
    end
    
    
    if ~isempty(results.Boxes{i})
       RGB = insertShape(I,'Rectangle',Modtestresult.GBoxes{i},...
                                 'LineWidth',3,'Color','green');
       RGB = insertObjectAnnotation(RGB,'rectangle',results.Boxes{i},results.Labels{i},...
                                 'LineWidth',3,'Color','yellow','TextColor','black');
    else
       RGB = insertObjectAnnotation(I,'rectangle',Modtestresult.GBoxes{i},Modtestresult.GLabels{i},...
                                 'LineWidth',3,'Color','green','TextColor','black');
    end
    imshow(RGB);
end

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, preprocessedTestData);
recall = table(recall);
precision = table(precision);

% Plot precision-recall curve.
figure
subplot(221);plot(recall.recall{1,1}, precision.precision{1,1});xlabel('Recall');ylabel('Precision');title(sprintf('Average Precision "Cylinder" = %.2f', ap(1)));grid on
subplot(222);plot(recall.recall{2,1}, precision.precision{2,1});xlabel('Recall');ylabel('Precision');title(sprintf('Average Precision "Box" = %.2f', ap(2)));grid on
subplot(223);plot(recall.recall{3,1}, precision.precision{3,1});xlabel('Recall');ylabel('Precision');title(sprintf('Average Precision "Pyramid" = %.2f', ap(3)));grid on
subplot(224);plot(recall.recall{4,1}, precision.precision{4,1});xlabel('Recall');ylabel('Precision');title(sprintf('Average Precision "Traffic Cone" = %.2f', ap(4)));grid on


%% testing
% Read the datastore.
reset(preprocessedTestData)

for i = 1:numImages
    
data = read(preprocessedTestData);
% Get the image.
I = data{1};

% Convert to dlarray.
XTest = dlarray(I,'SSCB');

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    XTest = gpuArray(XTest);
end

[bboxes, scores, labels] = yolov3Detect(net, XTest, networkOutputs, anchorBoxes, anchorBoxMasks, confidenceThreshold, overlapThreshold, classNames);

% Display the detections on image.
if ~isempty(scores)
    format = 'Ctg: %s Pr: %0.2f';
    str = sprintf(format,labels,scores);
    label = {str};
    I = insertObjectAnnotation(I, 'rectangle', bboxes, label);
end
figure
imshow(I)
end
%Functions
% The helper functions this script calls (augmentData, preprocessData,
% squeezenetFeatureExtractor, addFirstDetectionHead, addSecondDetectionHead,
% configureTrainingProgressPlotter, createBatchData,
% piecewiseLearningRateWithWarmup, modelGradients, displayLossInfo and
% yolov3Detect) are in detection/yolov3/, rewritten in 2026.

function gTruth = createdatatable(DS)

n = size(DS.gTruth.ROILabelData.video_20201004_124042);
image_filename = string.empty(n(1),0);
for i = 1:n(1)
    image_filename(i,1) = string(['Image_data/Image' int2str(i), '.jpg']);
end
gTruth(:,1) = table(image_filename);
tb = timetable2table(DS.gTruth.ROILabelData.video_20201004_124042(:,1:n(2)));
gTruth(:,2:n(2)+1) = tb(:,2:n(2)+1);

end

function gTruth = float2int(gTruth)

if istable(gTruth)
  varname=gTruth.Properties.VariableNames;
end

N = 1;

if N == 1
    gTruth(:,N) = gTruth(:,1);
end
    
for i = 2:size(gTruth,2)
    col = gTruth(:,i);
    
    for ii = 1:size(gTruth,1)
        bboxs = col(ii,1);
        bboxs = bboxs.(char(varname{1,i})){1,1};    
        Nboxs = size(bboxs,1);
        
        for iii = 1:Nboxs
            bbox = bboxs(iii,:);
            
            for iiii = 1:size(bbox,2)
                    if bbox(1,iiii) <= 1
                       bbox(1,iiii) = 1; 
                    else
                        bbox(1,iiii) = ceil(bbox(1,iiii));
                    end
            end
            bboxs(iii,:) = bbox; 
        end
      
        Col(ii,1) = {bboxs}; 
    
    end
    
    gTruth(:,i) = Col;
   
end

title = {'Image_FileName', 'Cone', 'Cup', 'Cube', 'Cylinder', 'Pyramid'};
gTruth.Properties.VariableNames = title;
end
