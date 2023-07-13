close all; clear all; clc;
%% Loading final random forest model.
load('rf_Mdl_optimised.mat')

%% Loading test sets for attributes, X, and target variable, Y.
load('test_set_partitions.mat','X_test','Y_test','X_train','Y_train')

%% Random Forest classification error:
%{
Calculating our classification error for our optimised Random Forest model, 
using the holdout split defined in the dataset_partitions.m file, by the 
variable cvp, testing on our test sets. 
%}
rf_err = loss(rf_Mdl_optimised,X_test,Y_test);

%% Random Forest training error:
%{
Calculating our training error for our optimised Random Forest model, 
using the holdout split defined in the dataset_partitions.m file, by the 
variable cvp, testing on our train sets.  
%}
rf_train_err = loss(rf_Mdl_optimised,X_train,Y_train);

%% Confusion matrix
rf_Y_pred = predict(rf_Mdl_optimised, X_test);
rf_confusion_matrix = confusionchart(Y_test, rf_Y_pred, 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
rf_cm_values = rf_confusion_matrix.NormalizedValues;
rf_confusion_matrix.Title = ({'Random Forest (Optimised Parameters):';'Confusion Matrix'});

%% Precision and Recall
[prec_mic,recall_mic,prec_mac,recall_mac,precision,recall,F1_mic,F1_mac] = multinomial_prec_recall(rf_cm_values);
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

%% Acquiring Out-of-Bag predictor importance estimates for attributes in X using out-of-bag samples
predictor_importance_est = oobPermutedPredictorImportance(rf_Mdl_optimised);
disp(predictor_importance_est);
% Plotting bar graph
figure
X_names = categorical({'Acousticness','Danceability','Energy','Instrumentalness','Liveness','Loudness','Mode','Popularity','Speechiness','Tempo','Valence'});
bar_graph = bar(X_names, predictor_importance_est)
title({'Out-of-Bag Permuted Predictor Importance Estimates';'for Attributes (X)'});
ylabel('Classification Error');

%% Defining functions
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