 % 
 %  \brief     Running on MNIST dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

% Running MNIST-LLE Experiments
clear all; close all; clc;

folder_prefix = 'Arcene_Experiments/';
mkdir(folder_prefix);

% Train Data-size
N_train = 150;
% Test Data-size
N_test = 200-N_train;

% Num of neighbors
K = [10; 12; 16; 20; 24; 32; 64];
% K = [8];

% Num of new dimensionality 
d = [10; 16; 20; 24; 32; 40; 52; 64; 96; 128];
% d = [10; 16; 20; 24; 32; 40; 52; 64; 96];

% Batch size (Must be grater thna K) 
% batch_size = [100; 200; 1000];
% batch_size = [1000; 2000; 10000];
batch_size = N_train;

% Debugging flags
make_data = 0;
plot_flag = 0;

%% Load Arcene Dataset
load('dataset/arcene_data.mat');
all_data = [arcene_train_data; arcene_valid_data];
all_labels = [arcene_train_labels; arcene_valid_labels];
rand_ids = randperm(length(all_labels))';
all_data = all_data(rand_ids,:);
all_labels = all_labels(rand_ids,:);

X = all_data(1:N_train,:)';
s2 = all_labels(1:N_train,1);
testX = all_data(N_train+1:length(all_labels),:)';
s2Test = all_labels(N_train+1:length(all_labels),1);

results_classification_acc = zeros(length(K), length(batch_size),length(d));

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
        
        Y = digits(X, testX, K(i,1), d(end,1), fid, N_train, batch_size(k,1), batch_folder_name);
        
        % Classification results
        tic;
	acc = 0;
        acc = Classification_with_DimRed(Y, s2, s2Test, N_train, N_test, fid);
	results_classification_acc(i,k,length(d)) = acc;
        fprintf(fid,'Classification without dimRed elapsed time: %6f \n',toc);
        fclose(fid);
        
        % Keep the lower dimensions
        for j=1:(length(d)-1)
            % folder_name = strcat(folder_prefix, int2str(K(i,1)), 'nn_', int2str(d(j,1)), 'd/');
            % mkdir(strcat(folder_name));
            acc = 0;  
            fprintf('Running for K:%d, d:%d, batch_size:%d \n', K(i,1), d(j,1), batch_size(k,1));
        
           %  batch_folder_name = strcat(folder_name, int2str(batch_size(k,1)), '_batch_size/');
           %  mkdir(batch_folder_name);
            
            file_name = strcat(batch_folder_name, int2str(K(i,1)), 'nn_', int2str(d(j,1)), 'd',int2str(batch_size(k,1)),'_batch.txt');
            fid=fopen(file_name,'w');
            
            lowerY = Y(1:d(j,1),:);
            
            % Classification results
            tic;
            acc = Classification_with_DimRed(lowerY, s2, s2Test, N_train, N_test, fid);
	    results_classification_acc(i,k,j) = acc;
            fprintf(fid,'Classification with dimRed elapsed time: %6f \n',toc);
            fclose(fid);
        end
    end
    fprintf('\n');
    
    file_name = strcat(folder_name, int2str(K(i,1)), 'nn_784_d', '.txt');
    fid=fopen(file_name,'w');
    
    tic;
    Classification_without_DimRed(X, testX, s2, s2Test, N_test, fid);
    fprintf(fid,'Classification without reduction elapsed time: %6f \n',toc);
    fclose(fid);
end

fprintf('\n\n END \n');
% save('results_classification_err.mat','results_classification_err');

save(strcat(folder_prefix,'results_classification_acc.mat'),'results_classification_acc');

Visualization;