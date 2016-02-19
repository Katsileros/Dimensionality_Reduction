 % 
 %  \brief     Make MNIST dataset. 2D-Images to vector
 %  \author    Katsileros Petros
 %  \date      24/12/2015
 %  \copyright 
 %

%% Create MNIST dataset

%% Train
load('dataset/images.mat');
data = images;
load('dataset/labels.mat');
clear images;

X = zeros(size(data,1)*size(data,2),size(data,3));

for i=1:size(data,3)
    for j=1:size(data,1)
        X(size(data,2)*(j-1)+1:size(data,2)*j,i) = data(j,:,i);
    end
end

clear data;
save('dataset/X.mat','X');

s2 = zeros(size(X,2),1);
for i=1:size(X,2)
    s2(i,1) = find(labels(i,:) == 1);
end
save('dataset/s2.mat','s2');

%% Testing
load('dataset/images_test.mat');
load('dataset/labels_test.mat');

testX = zeros(size(images_test,1)*size(images_test,2),size(images_test,3));

for i=1:size(images_test,3)
    for j=1:size(images_test,1)
        testX(size(images_test,2)*(j-1)+1:size(images_test,2)*j,i) = images_test(j,:,i);
    end
end

clear images_test;
save('dataset/testX.mat','testX');

s2Test = zeros(size(testX,2),1);
for i=1:size(testX,2)
    s2Test(i,1) = find(labels_test(i,:) == 1);
end
save('dataset/s2Test.mat','s2Test');

