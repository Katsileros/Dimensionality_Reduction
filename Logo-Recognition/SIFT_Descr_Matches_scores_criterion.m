% 
%  \brief     Matching features between testing and training logos
%             using as criterion the minimum score.
%  \author    Katsileros Petros
%  \date      18/12/2015
%  \copyright 
%

matches = cell( length(testingImagesFiles),length(trainingImagesFiles) );
scores = cell( length(testingImagesFiles),length(trainingImagesFiles) );
numMatches = zeros( length(testingImagesFiles),length(trainingImagesFiles) );

classify_labels = cell(length(testingImagesFiles),1);

classification_acc = length(testingImagesFiles);
 mean_average_scores = zeros(length(testingImagesFiles),1);

fprintf('Running Classification (scores criterion) \n');
tic;

%% Matches between test and train
for i=1:length(testingImagesFiles)
%     fprintf('Running classification for test-logo: %d \n',i);
    
    tmp_parfor_test_SIFT_descriptors = test_SIFT_descriptors{i,1};
%     tic;
    parfor j=1:length(trainingImagesFiles)
        [matches{i,j}, scores{i,j}] = vl_ubcmatch(tmp_parfor_test_SIFT_descriptors,train_SIFT_descriptors{j,1}) ;
%         [matches{i,j}, scores{i,j}] = vl_ubcmatch(train_SIFT_descriptors{j,1},test_SIFT_descriptors{i,1});
%         [matches{i,j}, scores{i,j}] = vl_ubcmatch(train_SIFT_descriptors{j,1},tmp_parfor_test_SIFT_descriptors) ;
        numMatches(i,j) = size(matches{i,j},2) ;
        mean_average_scores(i,j) = sum(scores{i,j});
    end
    
%     toc;
    mean_average_scores(i,numMatches(i,:) == 0) = 0;
    [tmp1, tmp2] = min(mean_average_scores(i,mean_average_scores(i,:) > 0));
    
    if(~isempty(tmp1))
        min_ids = find(mean_average_scores(i,:) == tmp1);
        
        %     tic;
        if(length(min_ids) > 1)
            % There are more than 1 matches with max-matches
            % Choose the best, according to their scores
            
            best = 0;
            final_id = 0;
            for j=1:length(min_ids)
                
                if(numMatches(i,min_ids(1,j)) > best)
                    best = numMatches(i,min_ids(1,j));
                    final_id = min_ids(1,j);
                end
                
                %% DEBUGGING
                enable = 0;
                if(enable)
                    X1 = train_SIFT_features{j,1}(1:2,matches{i,j}(1,:)) ; X1(3,:) = 1 ;
                    X2 = test_SIFT_features{i,1}(1:2,matches{i,j}(2,:)) ; X2(3,:) = 1 ;
                    
                    % --------------------------------------------------------------------
                    %                                                         Show matches
                    % --------------------------------------------------------------------
                    
                    dh1 = max(size(testingImagesData{i,1},1)-size(trainingImagesData{j,1},1),0) ;
                    dh2 = max(size(trainingImagesData{j,1},1)-size(testingImagesData{i,1},1),0) ;
                    
                    figure(1) ; clf ;
                    imshow([padarray(trainingImagesData{j,1},dh1,'post') padarray(testingImagesData{i,1},dh2,'post')]) ;
                    o = size(trainingImagesData{j,1},2) ;
                    line([train_SIFT_features{j,1}(1,matches{i,j}(1,:));test_SIFT_features{i,1}(1,matches{i,j}(2,:))+o], ...
                        [train_SIFT_features{j,1}(2,matches{i,j}(1,:));test_SIFT_features{i,1}(2,matches{i,j}(2,:))]) ;
                    title(sprintf('%d matches', tmp1)) ;
                    axis image off ;
                    
                end
                
            end
        else
            final_id = tmp2;
        end
        %     toc;
        
        classify_labels{i,1} = trainingImagesLogos{final_id,1};
        
        if(~strcmp(classify_labels{i,1}, testingImagesLogos{i,1}))
            classification_acc = classification_acc - 1;
        else
            enable = 0;
            if(enable)
                figure(1);
                subplot(1,2,1);imshow(testingImagesData{i,1});
                subplot(1,2,2);imshow(trainingImagesData{final_id,1});
                
                %% Features plot (DEBUGGING)
                X1 = train_SIFT_features{final_id,1}(1:2,matches{i,final_id}(1,:)) ; X1(3,:) = 1 ;
                X2 = test_SIFT_features{i,1}(1:2,matches{i,final_id}(2,:)) ; X2(3,:) = 1 ;
                
                % --------------------------------------------------------------------
                %                                                         Show matches
                % --------------------------------------------------------------------
                
                dh1 = max(size(testingImagesData{i,1},1)-size(trainingImagesData{final_id,1},1),0) ;
                dh2 = max(size(trainingImagesData{final_id,1},1)-size(testingImagesData{i,1},1),0) ;
                
                figure(2) ; clf ;
                imshow([padarray(trainingImagesData{final_id,1},dh1,'post') padarray(testingImagesData{i,1},dh2,'post')]) ;
                o = size(trainingImagesData{final_id,1},2) ;
                line([train_SIFT_features{final_id,1}(1,matches{i,final_id}(1,:));test_SIFT_features{i,1}(1,matches{i,final_id}(2,:))+o], ...
                    [train_SIFT_features{final_id,1}(2,matches{i,final_id}(1,:));test_SIFT_features{i,1}(2,matches{i,final_id}(2,:))]) ;
                title(sprintf('%d matches', tmp1)) ;
                axis image off ;
            end
        end
    else
        classification_acc = classification_acc - 1;
    end
    
end
toc;

fprintf('Accuracy: %f \n',(classification_acc./length(testingImagesFiles)).*100);


