close all; clear all; clc;
%% Loading dataset vairable.
load('VIF_feature_selection.mat')
%% Acquiring X and Y for VIF-reduced dataset:
%{
Isolating response variable, Y, before turning it into a categorical cell.
We then change the string class values into indices and, finally, make sure 
that Y is in the format of a categorical cell.
%}
Y = table2cell(music_data_clean(:,end));
Y = categorical(Y);
Y_numerical = grp2idx(Y);
Y_numerical = categorical(Y_numerical);

%{
The possible classes in Y have now been numerically encoded. In ascending
order from 1-6, we have the genres: 'Alternative', 'Blues', 'Classical', 
'Electronic', 'Hip-Hop' and 'Jazz').
%}

%Isolating attribute variables, X.
X = double(table2array(music_data_clean(:,1:end-1)));

%% Holdout train/test split:
rng('default'); %For reproducibility.
%{
Defining a stratified holdout partition, with our test set comprising of 
30% of our entire dataset. cvp contains the index sets of the datapoints 
included in our train and test sets.
%}
cvp = cvpartition(Y, 'Holdout', 0.3, 'Stratify', true);

%{
Using the index sets in cv to acquire our train and test sets for features,
X, and corresponding classes, Y.
%}
X_train = X(training(cvp),:);
Y_train = Y_numerical(training(cvp));
X_test = X(test(cvp),:);
Y_test = Y_numerical(test(cvp));

%% K-fold split:
num_folds = 10; %Number of folds in our k-fold split.
rng(1); %For reproducibility.
%{
Defining a stratified k-fold partition with k=10. cvp_kfold contains the 
index sets of the datapoints included in each of our k train and test folds
%}
cvp_kfold = cvpartition(Y_train,'KFold',num_folds,'Stratify',true);

%% Saving variables in .mat file
save dataset_partitions_VIF.mat