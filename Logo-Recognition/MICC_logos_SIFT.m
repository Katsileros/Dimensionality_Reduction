% 
%  \brief     MICC Classification using SIFT features
%  \author    Katsileros Petros
%  \date      27/12/2015
%  \copyright 
%

% SIFT feature extraction
clear all; close all; clc;

% Add VLFeat library to path
    addpath('vlfeat-0.9.20/');
    addpath('vlfeat-0.9.20/toolbox');
% Setup VLFeat
    vl_setup;

% logos folders
folders = {'agip';'apple';'barilla';'birra_moretti';'cinzano';'cocacola';'esso';'ferrari';'heineken'; ...
           'marlboro';'mcdonalds';'pepsi';'starbucks'};

% Train and test data size
Ntrain = length(folders)*5;
Ntest = 720;
Ntest_Classes = [15; 52; 49; 30; 67; 78; 86; 63; 49; 52; 52; 40; 87];


%% Train
tic;

trainingImagesFiles = cell(Ntrain,1);
trainingImagesLogos = cell(Ntrain*5,1);
% Read images and logo-labels
for i=1:length(folders)
    % Initial Class-i logo
    trainingImagesFiles{(i-1)*5 + 1,1} = strcat('MICC-Logos/',folders{i,1},'/reference_logo/',folders{i,1},'_logo.png');
    trainingImagesLogos{(i-1)*5 + 1,1} = folders{i,1};
    for j=2:5
        trainingImagesFiles{(i-1)*5 + j,1} = strcat('MICC-Logos/',folders{i,1},'/reference_logo/', ...
                                                               folders{i,1},'_logo_affine_warped_',int2str(j-1),'.png');
        trainingImagesLogos{(i-1)*5 + j,1} = folders{i,1};
    end
end

fprintf('Running SIFT-Feature extraction to %d training logos \n', Ntrain);

% Load images
trainingImagesData  = cell(Ntrain,1);
train_SIFT_features = cell(Ntrain,1);
train_SIFT_descriptors = cell(Ntrain,1);

for i=1:Ntrain
%     fprintf('Running SIFT-Feature extraction to training image - no: %d \n',i);
    
    tmp_image = imread(trainingImagesFiles{i,1});
    
    % Crop the logo Bounding Box and keep 1-channel (grayscale)
    trainingImagesData{i,1} = tmp_image(:,:,1);
    
    % Make single (for SIFT)
    trainingImagesData{i,1} = im2single(trainingImagesData{i,1});
    
    % SIFT Features Compute
    [train_SIFT_features{i,1},train_SIFT_descriptors{i,1}] = vl_sift(trainingImagesData{i,1}) ;
end
toc;


%% Testing
tic;

testingImagesFiles = cell(Ntest,1);
testingImagesLogos = cell(Ntest, 1);

counter = 1;
% Read images and logo-labels
for i=1:length(folders)
    for j=1:Ntest_Classes(i,1)
        testingImagesFiles{counter,1} = strcat('MICC-Logos/',folders{i,1},'/',folders{i,1},'_',int2str(j),'.png');
        testingImagesLogos{counter,1} = folders{i,1};
        counter = counter + 1;
    end
end

fprintf('Running SIFT-Feature extraction to %d testing logos \n', Ntest);

% Allocate SIFT arrays
testingImagesData  = cell(Ntest,1);
test_SIFT_features = cell(Ntest,1);
test_SIFT_descriptors = cell(Ntest,1);

for i=1:Ntest
%     fprintf('Running SIFT-Feature extraction to testing image - no: %d \n',i);
        
   tmp_image = imread(testingImagesFiles{i,1});
    
    % Crop the logo Bounding Box and keep 1-channel (grayscale)
    testingImagesData{i,1} = tmp_image(:,:,1);
    
    % Make single (for SIFT)
    testingImagesData{i,1} = im2single(testingImagesData{i,1});
    
    % SIFT Features Compute
    [test_SIFT_features{i,1},test_SIFT_descriptors{i,1}] = vl_sift(testingImagesData{i,1}) ;
end
toc;

%% Matches between test and train
SIFT_Descr_Matches_matches_criterion;

