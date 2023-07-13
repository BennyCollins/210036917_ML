close all; clear all; clc;
%% Loading partitions of dataset
load('dataset_partitions.mat')

%{
The possible classes in Y are: 'Alternative', 'Blues', 'Classical', 
'Electronic', 'Hip-Hop' and 'Jazz'.
%}

%% Multinomial Multiple Logistic Regression K-Fold Cross-Validation Classification Error
%{
Calculating our k-fold cross-validation classification error for 
Multinomial Multiple Logistic Regression, using the k-fold split defined 
in dataset_partitions.mat and assigned to the variable cvp_kfold. 
The mnr_classf function is defined at the end of the file, with 
accompanying explanatory comments.
%}
tic
mnr_kfold_err = crossval('mcr', X_train, Y_train, 'Predfun', @mnr_classf, 'Partition', cvp_kfold);
toc

%% Multinomial Multiple Logistic Regression Classification Error
%{
Calculating our Multinomial Multiple Logistic Regression model's 
classification error, using the holdout split defined in 
dataset_partitions.mat and assigned to the variable cvp. 
This model is fit on X_train and Y_train and then tested on X_test and 
Y_test.
%}
tic
mnr_err = crossval('mcr', X, Y_numerical, 'Predfun', @mnr_classf, 'Partition', cvp);
toc

%% Multinomial Multiple Logistic Regression Training Classification Error
%{
Calculating our Multinomial Multiple Logistic Regression model's training
set classification error, using the holdout split defined in 
dataset_partitions.mat and assigned to the variable cvp. 
This model is fit on X_train and Y_train and then tested on X_train and 
Y_train.
%}
mnr_train_err = mnr_train_misclass(X_train, Y_train);

%% Multinomial Multiple Logistic Regression Coefficients and P-Values
%{
Acquiring our Multinomial Multiple Logistic Regression model's attribute 
coefficients. The model is fit on X_train and Y_train.
mnr_coeffs will be a matrix containing the coefficient estimates for the 
Multiple Logistic Regression model we have trained. mnr_coeffs will contain 
5 columns, listing the coefficients of each of the 5 other categories 
(1:'Alternative', 2:'Blues', 3:'Classical', 4:'Electronic' and 
6:'Jazz') individually fit against the baseline category, 5:'Hip-Hop', in a 
Binomial Logistic Regression model.
%}
tic
[mnr_Mdl,dev,stats] = mnrfit(X_train, Y_train);
toc
disp('The log coefficients are:');
disp(mnr_Mdl);
disp('The coefficients are:');
mnr_coeffs = exp(mnr_Mdl);
disp(mnr_coeffs);
disp('The p-values for the coefficients are:');
disp(stats.p);

%% Multinomial Multiple Logistic Regression Confusion Matrix
%{
Acquiring our Multinomial Multiple Logistic Regression model's confusion 
matrix. The model is fit on X_train and Y_train and tested on X_test, using
the mnr_classf function.
We then use the resulting predicted classes (mnr_Y_pred) along with the 
true class values (Y_test) to acquire our confusion matrix.
%}
mnr_Y_pred = mnr_classf(X_train, Y_train, X_test);
mnr_confusion_matrix = confusionchart(Y_test, mnr_Y_pred, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
mnr_cm_values = mnr_confusion_matrix.NormalizedValues;
mnr_confusion_matrix.Title = 'Multinomial Multiple Logistic Regression: Confusion Matrix';

%% Multinomial Multiple Logistic Regression Training Set Confusion Matrix
%{
Acquiring a the classification error and confusion matrix for a model that 
is tested on the same data it's trained on. This is in order to check for 
overfitting.
%}
mnr_Y_train_pred = mnr_classf(X_train, Y_train, X_train);
mnr_train_confusion_matrix = confusionchart(Y_train, mnr_Y_train_pred, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
mnr_train_confusion_matrix.Title = 'Multinomial Multiple Logistic Regression: Training set Confusion Matrix';

%% Precision and Recall
[prec_mic,recall_mic,prec_mac,recall_mac,precision,recall,F1_mic, F1_mac] = multinomial_prec_recall(mnr_cm_values);
disp('The micro-average precision was:');
disp(prec_mic);
disp('The micro-average recall was:');
disp(recall_mic);
disp('The micro-average F1 score was:');
disp(F1_mic);
disp('The macro-average precision was:');
disp(prec_mac);
disp('The macro-average recall was:');
disp(recall_mac);
disp('The macro-average F1 score was:');
disp(F1_mac);

%% Saving the variables in .mat file, as well as a file for the final model
save multiple_multinomial_logistic_regression.mat
save('final_models/mnr_Mdl.mat','mnr_Mdl', 'dev', 'stats')

%% Defining functions
%{
Defining a function, mnr_classf, to fit a Multinomial Multiple Logistic
Regression model using attributes and corresponding classes from the input 
training set arguments (X_train and Y_train, respectively), then using the 
corresponding test set attribute values (X_test) to predict our Y variable 
classifications for each individual collection of attributes.
%}
function mnr_pred = mnr_classf(X_train, Y_train, X_test)
    mdl = mnrfit(X_train, Y_train); %Fitting model on training set
    %{
    Acquiring matrix (yfit) where each row contains the probabilities of a 
    datapoint belonging to each individual class in Y. The columns 
    represent each different class.
    %}
    yfit = mnrval(mdl, X_test);
    %{
    Find the index of the maximum probability value in each row of yfit.
    %}
    [~,idx] = max(yfit,[],2);
    %{
    Use each index to represent our final classification value for each
    datapoint. The rows of pred will represent each datapoint.
    %}
    mnr_pred = categorical(idx);
    disp('Fold fit');
end 

%{
Defining a function, mnr_train_classf, which fulfils the same function as 
mnr_classf, however it uses the attribute train set (X_train) to predict 
our Y variable classifications, and then calculates the missclassification 
rate by comparing our training set classes with the predicted classes.
%}
function mnr_train_err = mnr_train_misclass(X_train, Y_train)
    mdl = mnrfit(X_train, Y_train); %Fitting model on training set
    %{
    Acquiring matrix (yfit) where each row contains the probabilities of a 
    datapoint belonging to each individual class in Y. The columns 
    represent each different class.
    %}
    yfit = mnrval(mdl, X_train);
    %{
    Find the index of the maximum probability value in each row of yfit.
    %}
    [~,idx] = max(yfit,[],2);
    %{
    Use each index to represent our final classification value for each
    datapoint. The rows of mnr_pred will represent each datapoint.
    %}
    mnr_pred = categorical(idx);
    % Obtaining matrix of binary values (1 = correct prediction, 0 =
    % incorrect)
    correct_pred = mnr_pred==Y_train;
    % Finding number of correct predictions
    num_corr_pred = sum(correct_pred);
    % Finding missclassification error
    mnr_train_err = 1-(num_corr_pred/length(correct_pred));
end 

%{ 
Defining function, mnr_prec_recall, to take a set of multiclassification 
confusion matrix values, cm_values, and return both micro and macro 
precision, recall and F1 scores as well as the individual precision and 
recall scores for each different class.
%}
function [prec_mic,recall_mic,prec_mac,recall_mac, precision, recall, F1_mic, F1_mac] = multinomial_prec_recall(cm_values)
    % Finding number of classes in multiclassification confusion matrix
    num_classes = length(cm_values(1,:));
    % Setting value of each component to 0, so they can be iteratively
    % summed
    sum_TP = 0;
    precision_micro_denom = 0;
    recall_micro_denom = 0;
    precision_macro_num = 0;
    recall_macro_num = 0;
    % Iterate over each different class
    for i = 1:num_classes
        cm_diag = diag(cm_values);
        TP{i} = cm_diag(i); % True positive value for class i
        FP{i} = sum(cm_values(:,i))-cm_diag(i); % False positive value for class i
        FN{i} = sum(cm_values(i,:))-cm_diag(i); % False negative value for class i
        precision_denom{i} = TP{i} + FP{i}; % Precision denominator for class i
        recall_denom{i} = TP{i} + FN{i}; % Recall denominator for class i
        precision{i} = TP{i}/precision_denom{i}; % Precision for class i
        recall{i} = TP{i}/recall_denom{i}; % Recall for class i
        sum_TP = sum_TP + TP{i}; % Iteratively summing TP
        % Iteratively summing relevant components to acquire components of
        % micro and macro-average precision and recall.
        precision_micro_denom = precision_micro_denom + precision_denom{i};
        recall_micro_denom = recall_micro_denom + recall_denom{i};
        precision_macro_num = precision_macro_num + precision{i};
        recall_macro_num = recall_macro_num + recall{i};
    end
    % Deriving final macro and micro scores
    prec_mic = sum_TP/precision_micro_denom;
    recall_mic = sum_TP/recall_micro_denom;
    prec_mac = precision_macro_num/num_classes;
    recall_mac = recall_macro_num/num_classes;
    F1_mic = (2*prec_mic*recall_mic)/(prec_mic+recall_mic);
    F1_mac = (2*prec_mac*recall_mac)/(prec_mac+recall_mac);
end