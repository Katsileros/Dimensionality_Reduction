 % 
 %  \brief     Running on MNIST dataset, LLE algorithm
 %  \author    Katsileros Petros
 %  \date      09/01/2016
 %  \copyright 
 %

% Running MNIST-LLE Experiments
clear all; close all; clc;

folder_prefix = 'MNIST_Experiments_Knn/';
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
%K = [8];

% Num of new dimensionality 
d = [10; 16; 20; 24; 32; 40; 52; 64; 96; 128; 164; 196; 256];
%d = [128];

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

% K-means clustering
nn = [1; 2];
%nn = 4;

results_classification_err = zeros(length(K), length(nn), length(d));

tmpNN = knnsearch(X',X','K',max(nn)+1);

for n=1:length(nn)

% Load initial data
load('dataset/X.mat');
X = X(:,1:N_train);
load('dataset/s2.mat');
s2 = s2(1:N_train,1);

tmp_Xids = [];
for i=1:size(X,2)
   tmp_Xids = [tmp_Xids tmpNN(i,2:nn(n,1)+1)]; 
end

final_Xids = unique(tmp_Xids','rows');
clear tmp_Xids;

X = X(:,final_Xids);
s2 = s2(final_Xids,1);
Ntrain = length(final_Xids);
clear final_Xids;

% Iter for NN, d and batch_size
for i=1:length(K)
    folder_name = strcat(folder_prefix,int2str(nn(n,1)),'Knn/' , int2str(K(i,1)), 'nn_', int2str(d(end,1)), 'd/');
    mkdir(strcat(folder_name));
    
    fprintf('Running for K:%d, d:%d \n', K(i,1), d(end,1));
    
    batch_folder_name = strcat(folder_name, int2str(Ntrain), '_Ntrain/');
    mkdir(batch_folder_name);
    
    file_name = strcat(batch_folder_name, int2str(K(i,1)), 'nn_', int2str(d(end,1)), 'd',int2str(Ntrain),'_Ntrain.txt');
    fid=fopen(file_name,'w');
    
    Y = digits(X, testX, K(i,1), d(end,1), fid, batch_folder_name);
    
    % Classification results
    tic;
    err = 0;
    err = Classification_with_DimRed(Y, s2, s2Test, Ntrain, N_test, fid);
    results_classification_err(i,n,length(d)) = err;
    fprintf(fid,'Classification without dimRed elapsed time: %6f \n',toc);
    fclose(fid);
    
    % Keep the lower dimensions
    for j=1:(length(d)-1)
        err = 0;
        fprintf('Running for K:%d, d:%d, Knn:%d \n', K(i,1), d(j,1), nn(n,1));
        
        file_name = strcat(batch_folder_name, int2str(K(i,1)), 'nn_', int2str(d(j,1)), 'd',int2str(Ntrain),'_Ntrain.txt');
        fid=fopen(file_name,'w');
        
        lowerY = Y(1:d(j,1),:);
        
        % Classification results
        tic;
        err = Classification_with_DimRed(lowerY, s2, s2Test, Ntrain, N_test, fid);
        results_classification_err(i,n,j) = err;
        fprintf(fid,'Classification with dimRed elapsed time: %6f \n',toc);
        fclose(fid);
    end
   
    fprintf('\n');
    
    file_name = strcat(folder_name, int2str(K(i,1)), 'nn_784_d', '.txt');
    fid=fopen(file_name,'w');
    
    tic;
    Classification_without_DimRed(X, testX, s2, s2Test, N_test, fid);
    fprintf(fid,'Classification without reduction elapsed time: %6f \n',toc);
    fclose(fid);
end

end

fprintf('\n\n END \n');

save(strcat(folder_prefix,'/results_classification_err.mat'),'results_classification_err');

fid = fopen(strcat(folder_prefix,'time2.txt'),'w');
fprintf(fid,'End \n %s \n',datestr(now));
fclose(fid);

