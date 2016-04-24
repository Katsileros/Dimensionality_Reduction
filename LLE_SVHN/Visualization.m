%% Visualize Experiment results

clear all; close all; clc;
% Param: [K batch d]

% [2x2] kernel
% load('SVHN_LLE_Experiments/results_classification_err_kern2x2');
% no_dimRed = 20.25;

% [4x4] kernel
load('SVHN_LLE_Experiments/results_classification_err_kern4x4');
no_dimRed = 18.33;

% [8x8] kernel
% load('SVHN_LLE_Experiments/results_classification_err_kern8x8');
% no_dimRed = 21.52;

% Experiment parameters
% LLE Num of neighbors
neighbors = [8; 10; 12];

% LLE Num of new dimensionality 
dim = [16; 20; 32; 64; 96; 128; 164; 196; 256];

% Batch size (Must be grater thna K) 
% batch = [10000; 20000; 60000];
batch = 20000;

% Best accuracy
min = 100.00;
param = zeros(3,1);

% Loop for each dimension
for d=1:size(results_classification_err,3)
    figure(d);
    
    % Loop for each batch_size
    for b=1:size(results_classification_err,2)
        subplot(1,size(batch,1),b);
        
        % Loop for each neighbor
        for k=1:size(results_classification_err,1)
            if(results_classification_err(k,b,d) < min)
               min = results_classification_err(k,b,d);
               param(1,1) = k; param(2,1) = b; param(3,1) = d;
            end
            
            scatter(neighbors(k,1),double(results_classification_err(k,b,d)));hold on;
            text(neighbors(k,1), results_classification_err(k,b,d),num2str(results_classification_err(k,b,d)));
        end
        line([0,max(neighbors)+15],[no_dimRed,no_dimRed],'Color','r','LineWidth',2)
        title(strcat('dim-',int2str(dim(d,1)),',','batch-',int2str(batch(b,1))));
        xlabel('Number of nearest neighbors');
        ylabel('Mean average error');
        hold off;
    end
end

fprintf('Mean average error with dimensionality reduction: %f, dim: %d, batch: %d, neighbors: %d \n',...
        min, dim(param(1,1),1), batch(param(2,1),1), neighbors(param(3,1),1));
fprintf('Mean average error without dimensionality reduction: %f \n', no_dimRed);


