clear all; close all; clc;

load('dataset/X.mat');
load('dataset/testX.mat');
load('dataset/train_labels.mat');
load('dataset/test_labels.mat');

N_train = size(X,2);
N_train = 1000;
X = X(:,1:N_train);
s2 = train_labels(1:N_train,1);
N_test = size(testX,2);
N_test = 200;
testX = testX(:,1:N_test);
s2Test = test_labels(1:N_test,1);

% Nearest neighbors search
numNeighbors = 96;
err = zeros(numNeighbors,1);

 % tic;
[IDX1,~] = knnsearch(X',testX','K',numNeighbors);

% IDX2 = gpu_knn(double(X),double(testX),numNeighbors,1);
% IDX2 = IDX2';

% [~,idxHS,~] = gpuknnHeap(double(testX), double(X), numNeighbors, 1);

% keyboard;

%  toc;
 %size(IDX)
for nn=1:numNeighbors

IDX = IDX1(:,1:nn);

% MNIST labels
digit_labels = [1:10];
classification_labeling = zeros(1,N_test);
error_labels = zeros(1,10);
overall_digit_labels = zeros(1,10);

for i=1:N_test
    % Digit votings
    nn_votings = zeros(1,10);
    for k=1:nn
         k = (find(s2(IDX(i,k),1) == digit_labels));
         nn_votings(1,k) = nn_votings(1,k) + 1; 
    end
   
    [~,classification_labeling(1,i)] = max(nn_votings);
    
    overall_digit_labels(1,s2Test(i,1)) = overall_digit_labels(1,s2Test(i,1)) + 1;
    
    % Count accurate results
    if(classification_labeling(1,i) ~= s2Test(i,1))
        error_labels(1,s2Test(i,1)) = error_labels(1,s2Test(i,1)) + 1;
    end
end

for i=1:10
    error_labels(1,i) = error_labels(1,i) ./ overall_digit_labels(1,i);
    % fprintf('Error for digit-%d is: %f \n',mod(i,10),error_labels(1,i).*100);
end

err(nn,1) = (sum(error_labels) ./ 10 ) .* 100;
    
fprintf('\nMean average error (without dimensionality reduction): %f \n', err(nn,1));
end
