%
%   \brief     Simple K-means algorithm
%   \author    Katsileros Petros
%   \date      10/11/2015
%   \copyright 
%

function [c_coord, assignment] = simple_kmeans(X,k)
% X: input data [Nxd]
% d: data dimensionality
% k: number of nearest neighbors

% assignment[N x 1]: In which cluster each point belongs
% c_coords[k x d]: Coordinates of the final centroids

% fprintf('Running simple kmeans \n');

[N,d] = size(X);

% fprintf('Problem size: %d \n',N);

% Initial centroids
rng('default');
p = randperm(N);
p = 1:ceil(size(X,1) ./ k):size(X,1);
c_coord(1:k,:) = X(p(1,1:k),:);

max_iter = 10;
stop_crit = 1;
iter = 1;

% Loop until convergence
while( (stop_crit > 0.05) && (iter <= max_iter) )
%     fprintf('Iteration: %d,  ',iter);
    
    %% Very fast distance computation
    dst = bsxfun(@plus,(-2)*(c_coord*X'),dot(c_coord', c_coord',1)'); % compute the distances of all points to the clusters
    [~,assignment] = min(dst);
    clear dst;
    assignment = assignment';
    
    % Calculate new centroids
    old_c_coord = c_coord;
    
    % For every cluster, find new coords
    c_coord = zeros(k,d);
    for i=1:k
        if(length(find(assignment == i)) ~= 0)
            c_coord(i,:) = sum(X(find(assignment == i),:)) ./ length(find(assignment == i));
        end       
    end
    
    % Sum square error fro stopping criterion
    stop_crit = sum (sum( (c_coord - old_c_coord).^2 ) );
%     fprintf(' Sum square error: %f \n',stop_crit);
    
    % Iteration counter
    iter = iter + 1;
end

if(iter == max_iter)
    fprintf('Max iter(%d) for convergence \n',max_iter);
end

end
