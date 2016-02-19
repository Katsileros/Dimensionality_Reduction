 % 
 %  \brief     Running on MNIST dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

% Running MNIST-LLE Experiments
clear all; close all; clc;

folder_prefix = 'MNIST_Experiments_60000_Proj/';
mkdir(folder_prefix);

fid = fopen(strcat(folder_prefix,'time1.txt'),'w');
fprintf(fid,'Start \n %s \n',datestr(now));
fclose(fid);

% Train Data-size
N_train = 60000;
% Test Data-size
N_test = 10000;

% Num of neighbors
K = [8; 9; 10; 12; 16; 20];
% K = [12];

% Num of new dimensionality 
d = [10; 16; 20; 24; 32; 40; 52; 64; 96; 128; 164; 196; 256];
% d = [32; 48];
% d = 128;

% Batch size (Must be grater thna K) 
% batch_size = [100; 200; 1000];
batch_size = [10000; 20000; 60000];
% batch_size = 5000;

% Debugging flags
make_data = 0;
plot_flag = 0;

%% Create MNIST dataset

if(make_data)
    makeData;
else   
    load('dataset/X.mat');
    X = X(:,1:N_train);
    load('dataset/testX.mat');
    testX = testX(:,1:N_test);
    load('dataset/s2.mat');
    s2 = s2(1:N_train,1);
    load('dataset/s2Test.mat');
    s2Test = s2Test(1:N_test,1);
end

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
        
        Y = digits(X, K(i,1), d(end,1), fid, N_train, batch_size(k,1), batch_folder_name);
        
        % Classification results
        tic;
	err = 0;
        err = Classification_with_DimRed_Proj(Y, X, testX, s2, s2Test, N_train, N_test, batch_size(k,1), fid);
	results_classification_err(i,k,length(d)) = err;
        fprintf(fid,'Classification with dimRed elapsed time: %6f \n',toc);
        fclose(fid);
        
        % Keep the lower dimensions
        for j=1:(length(d)-1)
            % folder_name = strcat(folder_prefix, int2str(K(i,1)), 'nn_', int2str(d(j,1)), 'd/');
            % mkdir(strcat(folder_name));
            err = 0;  
            fprintf('Running for K:%d, d:%d, batch_size:%d \n', K(i,1), d(j,1), batch_size(k,1));
        
           %  batch_folder_name = strcat(folder_name, int2str(batch_size(k,1)), '_batch_size/');
           %  mkdir(batch_folder_name);
            
            file_name = strcat(batch_folder_name, int2str(K(i,1)), 'nn_', int2str(d(j,1)), 'd',int2str(batch_size(k,1)),'_batch.txt');
            fid=fopen(file_name,'w');
            
            lowerY = Y(1:d(j,1),:);
            
            % Classification results
            tic;
            err = Classification_with_DimRed_Proj(lowerY, X, testX, s2, s2Test, N_train, N_test, batch_size(k,1), fid);
	    results_classification_err(i,k,j) = err;
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

save(strcat(folder_prefix,'results_classification_err.mat'),'results_classification_err');

fid = fopen(strcat(folder_prefix,'time2.txt'),'w');
fprintf(fid,'End \n %s \n',datestr(now));
fclose(fid);

