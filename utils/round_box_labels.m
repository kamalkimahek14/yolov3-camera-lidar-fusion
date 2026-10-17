% Original MS thesis code (2021), Kamal Mehta. See README.md.
clear all; clc;
% Number of Column of the dataset
gtruth = load('gtruth.mat');
gtruth = gtruth.Gtruth;
Ncol = size(gtruth,2);
N = 1;

if N == 1
    Gtruth(:,N) = gtruth(:,1);
    N = N+1;
end
    
if N == 2
    col = gtruth(:,N);
    
    for ii = 1:size(gtruth,1)
        bboxs = col(ii,1);
        bboxs = bboxs.Chair{1,1};        % Change is required
        Nboxs = size(bboxs,1);
      
        Bboxs = {int(bboxs)};
        Col(ii,1) = Bboxs; 
    
    end
    
    Gtruth(:,N) = Col;
    N = N + 1;
end

if N == 3
    col = gtruth(:,N);
    
    for ii = 1:size(gtruth,1)
        bboxs = col(ii,1);
        bboxs = bboxs.CPU{1,1};        % Change is required
        Nboxs = size(bboxs,1);
      
        Bboxs = {uint16(bboxs)};
        Col(ii,1) = Bboxs; 
    
    end
    
    Gtruth(:,N) = Col;
    N = N + 1;
end

if N == 4
    col = gtruth(:,N);
    
    for ii = 1:size(gtruth,1)
        bboxs = col(ii,1);
        bboxs = bboxs.Refrigerator{1,1};        % Change is required
        Nboxs = size(bboxs,1);
      
        Bboxs = {uint16(bboxs)};
        Col(ii,1) = Bboxs; 
    
    end
    
    Gtruth(:,N) = Col;
    N = N + 1;
end
title = {'ImageFileName','Chair','CPU','Refrigerator'};
Gtruth.Properties.VariableNames = title;
