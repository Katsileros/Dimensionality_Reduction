%% Visualize Experiment results

clear all; close all; clc;
% Param: [K clust d]
load('MNIST_Experiments_Kmeans/results_classification_err.mat');

no_dimRed = [4.4310; 4.0255; 3.6742; 3.5085];

% Experiment parameters
% LLE Num of neighbors
neighbors = [8; 9; 10; 12; 16; 20];

% LLE Num of new dimensionality 
dim = [10; 16; 20; 24; 32; 40; 52; 64; 96; 128; 164; 196; 256];

% Clust size  
clust = [5000; 10000; 15000; 20000];

% Best accuracy
min = 100.00;
param = zeros(3,1);

% Loop for each dimension
for d=1:size(results_classification_err,3)
    figure(d);
    
    % Loop for each clust_size
    for b=1:size(results_classification_err,2)
        subplot(size(clust,1),1,b);
        
        % Loop for each neighbor
        for k=1:size(results_classification_err,1)
            if(results_classification_err(k,b,d) < min)
               min = results_classification_err(k,b,d);
               param(1,1) = k; param(2,1) = b; param(3,1) = d;
            end
            
            scatter(neighbors(k,1),double(results_classification_err(k,b,d)));hold on;
            text(neighbors(k,1), results_classification_err(k,b,d),num2str(results_classification_err(k,b,d)));
        end
        line([0,max(neighbors)+15],[no_dimRed(b,1),no_dimRed(b,1)],'Color','r','LineWidth',2)
        title(strcat('dim-',int2str(dim(d,1)),',','clust-',int2str(clust(b,1))));
        xlabel('Number of nearest neighbors');
        ylabel('Accuracy error');
        hold off;
    end
end

fprintf('Best accuracy with dimensionality reduction: %f, dim: %d, clust: %d, neighbors: %d \n',...
        min, dim(param(3,1),1), clust(param(2,1),1), neighbors(param(1,1),1));
fprintf('Accuracy without dimensionality reduction: %f \n', no_dimRed);


