%% Visualize Experiment results

clear all; close all; clc;
% Param: [K batch d]
load('Arcene_Experiments/results_classification_acc');

no_dimRed = 73.33;

% Experiment parameters
% LLE Num of neighbors
neighbors = [10; 12; 16; 20; 24; 32; 64];

% LLE Num of new dimensionality 
dim = [10; 16; 20; 24; 32; 40; 52; 64; 96; 128];

% Batch size (Must be grater thna K) 
% batch = [10000; 20000; 60000];
batch = [150];

% Best accuracy
max_acc = 0;
param = zeros(3,1);

% Loop for each dimension
for d=1:size(results_classification_acc,3)
    figure(d);
    
    % Loop for each batch_size
    for b=1:size(results_classification_acc,2)
        subplot(1,size(batch,1),b);
        
        % Loop for each neighbor
        for k=1:size(results_classification_acc,1)
            if(results_classification_acc(k,b,d) > max_acc)
               max_acc = results_classification_acc(k,b,d);
               param(1,1) = k; param(2,1) = b; param(3,1) = d;
            end
            
            scatter(neighbors(k,1),double(results_classification_acc(k,b,d)));hold on;
            text(neighbors(k,1), results_classification_acc(k,b,d),num2str(results_classification_acc(k,b,d)));
        end
        line([0,max(neighbors)+15],[no_dimRed,no_dimRed],'Color','r','LineWidth',2)
        title(strcat('dim-',int2str(dim(d,1)),',','batch-',int2str(batch(b,1))));
        xlabel('Number of nearest neighbors');
        ylabel('Accuracy error');
        hold off;
    end
end

fprintf('Best accuracy with dimensionality reduction: %f, dim: %d, batch: %d, neighbors: %d \n',...
        max_acc, dim(param(1,1),1), batch(param(2,1),1), neighbors(param(3,1),1));
fprintf('Accuracy without dimensionality reduction: %f \n', no_dimRed);


