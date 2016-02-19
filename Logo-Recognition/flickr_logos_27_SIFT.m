% 
%  \brief     Flickr_logos_27 Classification using SIFT features
%  \author    Katsileros Petros
%  \date      18/12/2015
%  \copyright 
%

% SIFT feature extraction
clear all; close all; clc;

% Add VLFeat library to path
    addpath('vlfeat-0.9.20/');
    addpath('vlfeat-0.9.20/toolbox');
% Setup VLFeat
    vl_setup;

% txt-files format
formatSpec = '%s %s %d %d %d %d %d %*[^\n]';

% Read images and logo-labels
[dataImagesFiles, dataImagesLogos, ~, posX1, posY1, posX2, posY2] = ...
    textread('flickr_logos_27_dataset/flickr_logos_27_dataset_training_set_annotation.txt',formatSpec); %#ok<DTXTRD>

random_ids = randperm(length(dataImagesFiles))';

% Train and test data size
Ntest = length(dataImagesFiles);
% Ntrain = 100;
Ntrain = length(dataImagesFiles);

test_ids = random_ids(ceil( rand(1,Ntest) * length(dataImagesFiles) ),1)';
test_ids = test_ids(1,1:500);

% Random permute data
dataImagesFiles = dataImagesFiles(random_ids,1);
dataImagesLogos = dataImagesLogos(random_ids,1);
posX1 = posX1(random_ids,1);
posX2 = posX2(random_ids,1);
posY1 = posY1(random_ids,1);
posY2 = posY2(random_ids,1);

% Start from index 1, not zero.
posX1(posX1 == 0) = 1;
posX2(posX2 == 0) = 1;
posY1(posY1 == 0) = 1;
posY2(posY2 == 0) = 1;

%% Train
tic;

trainingImagesFiles = dataImagesFiles(1:Ntrain,1);
trainingImagesLogos = dataImagesLogos(1:Ntrain,1);

fprintf('Running SIFT-Feature extraction to %d training logos \n', length(trainingImagesFiles));

% Load images
trainingImagesData  = cell(length(trainingImagesFiles),1);
train_SIFT_features = cell(length(trainingImagesFiles),1);
train_SIFT_descriptors = cell(length(trainingImagesFiles),1);

parfor i=1:length(trainingImagesFiles)
%     fprintf('Running SIFT-Feature extraction to training image - no: %d \n',i);
    
    im_file = strcat('flickr_logos_27_dataset/flickr_logos_27_dataset_images/',trainingImagesFiles{i,1});
    tmp_image = imread(im_file);
    
    % Crop the logo Bounding Box and keep 1-channel (grayscale)
    trainingImagesData{i,1} = tmp_image(posY1(i,1):posY2(i,1),posX1(i,1):posX2(i,1),1);
    
    % Make single (for SIFT)
    trainingImagesData{i,1} = im2single(trainingImagesData{i,1});
    
    % SIFT Features Compute
    [train_SIFT_features{i,1},train_SIFT_descriptors{i,1}] = vl_sift(trainingImagesData{i,1}) ;
    
    %% LLE dimensionality reduction
%     train_SIFT_descriptors{i,1} = lle(train_SIFT_descriptors{i,1},4,32,1,'.');

end
toc;

%% Testing (with train images)
tic;

testingImagesFiles = dataImagesFiles(test_ids, 1);
testingImagesLogos = dataImagesLogos(test_ids, 1);

fprintf('Running SIFT-Feature extraction to %d testing logos \n', length(testingImagesFiles));

% Allocate SIFT arrays
testingImagesData  = cell(length(testingImagesFiles),1);
test_SIFT_features = cell(length(testingImagesFiles),1);
test_SIFT_descriptors = cell(length(testingImagesFiles),1);

% tmp variables for parfor
posX1 = posX1(test_ids,1);
posX2 = posX2(test_ids,1);
posY1 = posY1(test_ids,1);
posY2 = posY2(test_ids,1);

parfor i=1:length(testingImagesFiles)
%     fprintf('Running SIFT-Feature extraction to testing image - no: %d \n',i);
        
    im_file = strcat('flickr_logos_27_dataset/flickr_logos_27_dataset_images/',testingImagesFiles{i,1});
    tmp_image = imread(im_file);
    
    % Crop the logo Bounding Box and keep 1-channel (grayscale)
    testingImagesData{i,1} = tmp_image(posY1(i,1):posY2(i,1),posX1(i,1):posX2(i,1),1);
    
    % Make single (for SIFT)
    testingImagesData{i,1} = im2single(testingImagesData{i,1});
    
    % SIFT Features Compute
    [test_SIFT_features{i,1},test_SIFT_descriptors{i,1}] = vl_sift(testingImagesData{i,1}) ;
    
end

toc;

% %% Testing (with test images)
% tic;
% formatSpec = '%s %s %*[^\n]';
% % Read images and logo-labels
% [dataImagesFiles, dataImagesLogos] = textread('flickr_logos_27_dataset/flickr_logos_27_dataset_query_set_annotation.txt',formatSpec); %#ok<DTXTRD>
% 
% testingImagesFiles = dataImagesFiles(:, 1);
% testingImagesLogos = dataImagesLogos(:, 1);
% 
% fprintf('Running SIFT-Feature extraction to %d testing logos \n', length(testingImagesFiles));
% 
% % Allocate SIFT arrays
% testingImagesData  = cell(length(testingImagesFiles),1);
% test_SIFT_features = cell(length(testingImagesFiles),1);
% test_SIFT_descriptors = cell(length(testingImagesFiles),1);
% 
% parfor i=1:length(testingImagesFiles)
% %     fprintf('Running SIFT-Feature extraction to testing image - no: %d \n',i);
%         
%     im_file = strcat('flickr_logos_27_dataset/flickr_logos_27_dataset_images/',testingImagesFiles{i,1});
%     tmp_image = imread(im_file);
%     
%     % Crop the logo Bounding Box and keep 1-channel (grayscale)
% %     testingImagesData{i,1} = tmp_image;
%     
%     % Make single (for SIFT)
%     testingImagesData{i,1} = im2single(tmp_image(:,:,1));
%     
%     % SIFT Features Compute
%     [test_SIFT_features{i,1},test_SIFT_descriptors{i,1}] = vl_sift(testingImagesData{i,1}) ;
%     
% end
% 
% toc;

%% Matches between test and train

SIFT_Descr_Matches_matches_criterion;
% %SIFT_Descr_Matches_scores_criterion;

