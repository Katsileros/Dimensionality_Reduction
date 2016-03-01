 % 
 %  \brief     Running on SVHN dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      01/03/2016
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
d = [32];

% HoG kernel size 
hog_kernel = [4 4];
folder_prefix = strcat(folder_exp,'kern_size_[',int2str(hog_kernel(1,1)),'x',int2str(hog_kernel(1,2)),']/');

%% Load SVHN dataset

load('dataset/X.mat');
load('dataset/testX.mat');
load('dataset/train_labels.mat');
load('dataset/test_labels.mat');

%% Subsampling data
clust_ids = cell(10,1);
min_clust_size = intmax('int64');
for i=1:10
    clust_ids{i,1} = find(train_labels == i);
    if(size(clust_ids{i,1},1) < min_clust_size)
        min_clust_size = size(clust_ids{i,1},1);
    end
end

%clust_size = 1500;
clust_size = min_clust_size;
final_data = zeros(size(X,1),size(X,2),clust_size*10);
final_labels = zeros(clust_size*10,1);

num_sub_spaces = 3;
subspaces_length = floor(clust_size ./ num_sub_spaces);

for nsb=1:num_sub_spaces
    for i=1:10
        for j=1:subspaces_length
            final_data(:,:,(i-1)*(clust_size./num_sub_spaces) + (nsb-1)*(subspaces_length.*10) + j) = single(X(:,:,clust_ids{i,1}((nsb-1)*subspaces_length + j,1)));
            final_labels((i-1)*(clust_size./num_sub_spaces) + (nsb-1)*(subspaces_length.*10) + j,1) = train_labels(clust_ids{i,1}((nsb-1)*subspaces_length + j,1),1);

%%%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%
%             final_labels((i-1)*(clust_size./num_sub_spaces) + (nsb-1)*(subspaces_length.*10) + j,1) = (nsb-1)*subspaces_length + j;
%             final_labels((i-1)*(clust_size./num_sub_spaces) + (nsb-1)*(subspaces_length.*10) + j,1) = ...
%                          (i-1)*(clust_size./num_sub_spaces) + (nsb-1)*(subspaces_length.*10) + j;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
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

% Test Data-size
% N_test = size(testX,3);
N_test = 1000;
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
%     fprintf('Running Feature extraction to training image - no: %d \n',i);
    % HoG features
    train_descriptors{i,1} = extractHOGFeatures(X(:,:,i),'CellSize',hog_kernel);
    train_all_descriptors = [train_all_descriptors; train_descriptors{i,1}];
end

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
%     fprintf('Running Feature extraction to testing image - no: %d \n',i);
    % HoG Features
    test_descriptors{i,1} = extractHOGFeatures(testX(:,:,i),'CellSize',hog_kernel);
    test_all_descriptors = [test_all_descriptors; test_descriptors{i,1}];
end

[unique_test_all_descriptors,unique_test_desc,unique_test_bins] = unique(test_all_descriptors,'rows');          % HoG
unique_test_all_descriptors = double(unique_test_all_descriptors(unique_test_bins,:)');
test_all_descriptors = test_all_descriptors(unique_test_bins,:);
Ntest = size(unique_test_all_descriptors,2);

% Batch size 
% batch_size = floor([ Ntrain; (Ntrain./2); (Ntrain./4); (Ntrain./5) ]);
batch_size = floor([(Ntrain./num_sub_spaces)]);
% batch_size = Ntrain;

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
        
        Y = digits(unique_train_all_descriptors, unique_test_all_descriptors, K(i,1), d(end,1), fid, Ntrain, batch_size(k,1), batch_folder_name);
        Ntrain = size(unique_train_all_descriptors,2);
        Ntest = size(unique_test_all_descriptors,2);

        %% Make data-features to the appropriate format (Subspaces ids)
        % Train data
        tmp_train_all_desc = [];
        tmp_test_all_desc = [];
        for b=1:floor(Ntrain ./ batch_size(k,1))
            tmp_train_all_desc = [ tmp_train_all_desc Y(:,(b-1)*Ntest + ((b-1)*batch_size(k,1) + 1) : ...
                                                          (b-1)*Ntest + ((b-1)*batch_size(k,1) + batch_size(k,1)))];
            tmp_test_all_desc = [tmp_test_all_desc Y(:,(b-1)*(batch_size(k,1)+Ntest)+(batch_size(k,1)+(1:Ntest)))];                                          
        end

        Ntrain = size(tmp_train_all_desc,2);
        final_train_descr = cell(Ntrain,1);
        for t=1:Ntrain
            final_train_descr{t,1} = [];
            final_train_descr{t,1} = [final_train_descr{t,1} tmp_train_all_desc(:,t)] ; % HoG
        end
        
        % Test data
        Ntest = size(tmp_test_all_desc,2);
        final_test_descr = cell(N_test,1);
        for t=1:Ntest
            final_test_descr{t,1} = [];
            final_test_descr{t,1} = [final_test_descr{t,1} tmp_test_all_desc(:,t)] ; % HoG
        end
        
        % Classification results
        tic;
        err = 0;
        err = Classification_with_DimRed(final_train_descr, final_test_descr, train_labels, test_labels, Ntrain, Ntest, batch_size(k,1), d(end,1), fid, batch_folder_name) ;
        results_classification_err(i,k,length(d)) = err;
        fprintf(fid,'Classification without dimRed elapsed time: %6f \n',toc);
        fclose(fid);
        
        % Keep the lower dimensions
        for j=1:(length(d)-1)
            fprintf('Running for K:%d, d:%d, batch_size:%d \n', K(i,1), d(j,1), batch_size(k,1));
            
            file_name = strcat(batch_folder_name, int2str(K(i,1)), 'nn_', int2str(d(j,1)), 'd',int2str(batch_size(k,1)),'_batch.txt');
            fid=fopen(file_name,'w');
            
            % Classification results
            tic;
            err = 0;
            err = Classification_with_DimRed(final_train_descr, final_test_descr, train_labels, test_labels, N_train, N_test, batch_size(k,1), d(j,1), fid, batch_folder_name);
%             err = Classification_with_DimRed_swsto(final_train_descr, final_test_descr, train_labels, test_labels, N_train, N_test, fid);
            results_classification_err(i,k,j) = err;
            fprintf(fid,'Classification with dimRed elapsed time: %6f \n',toc);
            fclose(fid);
            
        end
    end
    fprintf('\n');
    
    file_name = strcat(folder_name, int2str(K(i,1)), 'nn_No_dimRed', '.txt');
    fid=fopen(file_name,'w');
    
    tic;
    
    err = Classification_without_DimRed(train_descriptors,test_descriptors, train_labels, test_labels, N_train, N_test, fid);
    fprintf(fid,'Classification without reduction elapsed time: %6f \n',toc);
    fclose(fid);
end

fprintf('\n\n END \n');
save(strcat(folder_prefix,'results_classification_err.mat'),'results_classification_err');


fid = fopen(strcat(folder_exp,'time2.txt'),'w');
fprintf(fid,'End \n %s \n',datestr(now));
fclose(fid);

