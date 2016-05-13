clear all;
close all;

load('arcene_train.data');
arcene_train_data = arcene_train;
save('arcene_train_data.mat','arcene_train_data');
clear all;

load('arcene_valid.data');
arcene_valid_data = arcene_valid;
save('arcene_valid_data.mat','arcene_valid_data');
clear all;

load('arcene_test.data');
arcene_test_data = arcene_test;
save('arcene_test_data.mat','arcene_test_data');
clear all;

load('arcene_train.labels');
arcene_train_labels = arcene_train;
save('arcene_train_labels.mat','arcene_train_labels');
clear all;

load('arcene_valid.labels');
arcene_valid_labels = arcene_valid;
save('arcene_valid_labels.mat','arcene_valid_labels');
clear all;
