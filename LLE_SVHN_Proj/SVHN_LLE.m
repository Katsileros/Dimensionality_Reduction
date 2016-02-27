 % 
 %  \brief     Running on SVHN dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      18/02/2016
 %  \copyright 
 %

% Running SVHN-LLE Experiments
clear all; close all; clc;

folder_exp = 'SVHN_LLE_Experiments/';
mkdir(folder_exp);

fid = fopen(strcat(folder_exp,'time1.txt'),'w');
fprintf(fid,'Start \n %s \n',datestr(now));
fclose(fid);

% Num of neighbors
% K = [8; 10; 12];
K = [12];

% Num of new dimensionality 
% d = [16; 20; 32; 64; 96; 128; 164; 196; 256]; % HoG
% d = [16; 20; 32; 64; 96]; % SIFT
d = [32];

% HoG kernel size 
% kernels = {[2 2], [4 4], [8 8]}; % HoG features size {8100,1764,324}
kernels = {[4 4]};
for kern=1:length(kernels)
hog_kernel = kernels{1,kern};
folder_prefix = strcat(folder_exp,'kern_size_[',int2str(hog_kernel(1,1)),'x',int2str(hog_kernel(1,2)),']/');

%% Load SVHN dataset

load('dataset/X.mat');
load('dataset/testX.mat');
load('dataset/train_labels.mat');
load('dataset/test_labels.mat');

%% Subsampling data
% Clust size is the number of train data for each class
clust_size = 300;
clust_ids = cell(10,1);
final_data = zeros(size(X,1),size(X,2),clust_size*10);
final_labels = zeros(clust_size*10,1);
for i=1:10
    clust_ids{i,1} = find(train_labels == i);
    tmp_rand_ids = randperm(length(clust_ids{i,1}))';
    for j=1:clust_size
        final_data(:,:,(i-1)*clust_size + j) = single(X(:,:,clust_ids{i,1}(tmp_rand_ids(j,1)),1));
        final_labels((i-1)*clust_size + j,1) = train_labels(clust_ids{i,1}(tmp_rand_ids(j,1),1),1);
    end
end

%  Train Data-size
clear X;
X = final_data;
clear final_data;
N_train = size(X,3);

clear train_labels;
train_labels = final_labels;
clear final_labels;

% %N_train = 1000;
% X = single(X(:,:,1:N_train));

% Test Data-size
% N_test = size(testX,3);
N_test = 200;
testX = single(testX(:,:,1:N_test));

%% Preprocessing
% Normalize data for better classification performance
X = X ./ 255;
testX = testX ./ 255;

fprintf('Train data feature extraction ... \n');

%% Apply Features on Train
train_features = cell(N_train,1);
train_descriptors = cell(N_train,1);
train_all_descriptors = [];
for i=1:N_train
%     fprintf('Running SIFT-Feature extraction to training image - no: %d \n',i);
    % HoG features
    train_descriptors{i,1} = extractHOGFeatures(X(:,:,i),'CellSize',hog_kernel);
    train_all_descriptors = [train_all_descriptors; train_descriptors{i,1}];
end

% [unique_train_all_descriptors,unique_train_desc,unique_train_bins] = unique(train_all_descriptors','rows');   % SIFT
[unique_train_all_descriptors,unique_train_desc,unique_train_bins] = unique(train_all_descriptors,'rows');      % HoG
unique_train_all_descriptors = double(unique_train_all_descriptors(unique_train_bins,:)');
train_all_descriptors = train_all_descriptors(unique_train_bins,:);
Ntrain = size(unique_train_all_descriptors,2);

fprintf('Test data feature extraction ... \n');
%% Apply Features on Test
test_features = cell(N_test,1);
test_descriptors = cell(N_test,1);
test_all_descriptors = [];
for i=1:N_test
%     fprintf('Running SIFT-Feature extraction to testing image - no: %d \n',i);
    % HoG Features
    test_descriptors{i,1} = extractHOGFeatures(testX(:,:,i),'CellSize',hog_kernel);
    test_all_descriptors = [test_all_descriptors; test_descriptors{i,1}];
end

% Batch size (Must be grater thna K) 
batch_size = Ntrain;

results_classification_err = zeros(length(K), length(batch_size),length(d));

% Iter for NN, d and batch_size
for i=1:length(K)
    folder_name = strcat(folder_prefix, int2str(K(i,1)), 'nn_', int2str(d(end,1)), 'd/');
    mkdir(strcat(folder_name));
    for k=1:length(batch_size)
        fprintf('Running for K:%d, d:%d, batch_size:%d \n', K(i,1), d(end,1), batch_size(k,1));
        
        batch_folder_name = strcat(folder_name, int2str(batch_size(k,1)), '_batch_size/');
        mkdir(batch_folder_name);
        
        file_name = strcat(batch_folder_name, int2str(K(i,1)), 'nn_', int2str(d(end,1)), 'd',int2str(batch_size(k,1)),'_batch.txt');
        fid=fopen(file_name,'w');
        
        Y = digits(unique_train_all_descriptors, K(i,1), d(end,1), fid);
        
        %% Make data-features to the appropriate format
        % Train data
        final_train_descr = cell(N_train,1);
        for t=1:N_train
            final_train_descr{t,1} = [];
            final_train_descr{t,1} = [final_train_descr{t,1} Y(:,t)] ; % HoG
        end
        
        % Classification results
        tic;
        err = 0;
        err = Classification_with_DimRed(unique_train_all_descriptors, train_descriptors, test_descriptors, train_labels, test_labels, N_train, N_test, fid) ;
%         err = Classification_without_DimRed_fake(final_train_descr, test_descriptors, train_labels, test_labels, N_train, N_test, fid);
        results_classification_err(i,k,length(d)) = err;
        fprintf(fid,'Classification without dimRed elapsed time: %6f \n',toc);
        fclose(fid);
        
    end
    fprintf('\n');
    
    file_name = strcat(folder_name, int2str(K(i,1)), 'nn_784_d', '.txt');
    fid=fopen(file_name,'w');
    
    tic;
    err = Classification_without_DimRed(train_descriptors,test_descriptors, train_labels, test_labels, N_train, N_test, fid);
    fprintf(fid,'Classification without reduction elapsed time: %6f \n',toc);
    fclose(fid);
end

fprintf('\n\n END \n');
save(strcat(folder_prefix,'results_classification_err.mat'),'results_classification_err');

end

fid = fopen(strcat(folder_exp,'time2.txt'),'w');
fprintf(fid,'End \n %s \n',datestr(now));
fclose(fid);

