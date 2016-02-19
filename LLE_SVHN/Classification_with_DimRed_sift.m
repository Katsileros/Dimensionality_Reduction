% 
%  \brief     Classification on reduced dimensionality 
%             MNIST dataset with NN algorithm
%  \author    Katsileros Petros
%  \date      06/02/2016
%  \copyright 
%

% Executing nearest neighbor search for classification results
function err = Classification_with_DimRed(train_SIFT_descr,test_SIFT_descr, train_labels, test_labels, N_train, N_test, fid)

decision_score = zeros( N_test, N_train );
classify_labels = cell(N_test,1);
classification_acc = N_test;

fprintf('Running Classification (numMatches criterion) \n');
tic;

%% Matches between test and train
for i=1:N_test
%     fprintf('Running classification for test-logo: %d \n',i);

    if(~isempty(test_SIFT_descr{i,1}))
        
%         tmp_parfor_test_SIFT_descriptors = test_SIFT_descr{i,1};
        for j=1:N_train
    %         [matches, scores] = vl_ubcmatch(tmp_parfor_test_SIFT_descriptors,train_SIFT_descriptors{j,1},0.8) ;
            if(~isempty(train_SIFT_descr{j,1}))
                [matches, scores] = vl_ubcmatch(train_SIFT_descr{j,1},test_SIFT_descr{i,1},1) ;
%                 decision_score(i,j) = sum(scores) / size(matches,2);
                decision_score(i,j) = sum(scores);
            else
                decision_score(i,j) = realmax('single');
            end
        end
    
        [~,final_id] = min(decision_score(i,:));  
        classify_labels{i,1} = train_labels(final_id,1);  

        if(classify_labels{i,1} ~= test_labels(i,1))
           classification_acc = classification_acc - 1; 
        end
    
    else
        classification_acc = classification_acc - 1;
    end
    
end
toc;
err = (classification_acc./N_test).*100;

fprintf(fid,'Accuracy: %f \n',(classification_acc./N_test).*100);
fprintf('Accuracy: %f \n',(classification_acc./N_test).*100);