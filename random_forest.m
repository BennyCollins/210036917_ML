close all; clear all; clc;
%% Loading partitions of dataset
load('final_model_partitions.mat')

%% Random Forests: Optimising Parameters
rng('default');
t = templateTree('Reproducible',true);
rf_Mdl = fitcensemble(X_train,Y_train,'Method','Bag','OptimizeHyperparameters',{'NumLearningCycles','MinLeafSize','NumVariablesToSample'},'Learners',t)
%% Creating variables for optimised hyperparameters
optimised_parameters = rf_Mdl.HyperparameterOptimizationResults.XAtMinObjective;
num_learning_cycles = optimised_parameters.NumLearningCycles;
min_leaf_size = optimised_parameters.MinLeafSize;
num_variables_sample = optimised_parameters.NumVariablesToSample;

%% Training Random Forest with optimised hyperparameters on k-folds
tic
rng('default');
t_optimised = templateTree('MinLeafSize',min_leaf_size,'NumVariablesToSample',num_variables_sample,'Reproducible',true);
rf_Mdl_optimised_kfold = fitcensemble(X_train,Y_train,'Method','Bag','NumLearningCycles',num_learning_cycles,'Learners',t_optimised,'CVPartition',cvp_kfold);
toc

%% Random Forest k-fold classification error:
%{
Calculating our k-fold cross validation classification error for our 
optimised Random Forest model, using the k-fold split defined in the 
dataset_partitions.m file, by the variable cvp_kfold. 
%}
tic
rf_kfold_err = kfoldLoss(rf_Mdl_optimised_kfold);
toc

%% Training Random Forest with optimised hyperparameters and training set
tic
rng('default');
rf_Mdl_optimised = fitcensemble(X_train,Y_train,'Method','Bag','NumLearningCycles',num_learning_cycles,'Learners',t_optimised);
toc

%% Random Forest classification error:
%{
Calculating our classification error for our optimised Random Forest model, 
using the holdout split defined in the dataset_partitions.m file, by the 
variable cvp, testing on our test sets. 
%}
tic
rf_err = loss(rf_Mdl_optimised,X_test,Y_test);
toc

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

%% Plot loss and k-fold loss over number of trees
figure
plot(loss(rf_Mdl_optimised,X_test,Y_test,'mode','cumulative'));
hold on
plot(kfoldLoss(rf_Mdl_optimised_kfold,'mode','cumulative'),'r');
hold off
title({'Random Forest with Optimised Hyperparameters:';'Classification Error vs Number of Trees'});
xlabel('Number of Trees');
ylabel('Classification Error');
legend('Test Set','K-Fold Cross-Validation','Location','NE');

%% Acquiring Out-of-Bag predictor importance estimates for attributes in X using out-of-bag samples
predictor_importance_est = oobPermutedPredictorImportance(rf_Mdl_optimised);
disp(predictor_importance_est);
% Plotting bar graph
figure
X_names = categorical({'Acousticness','Danceability','Energy','Instrumentalness','Liveness','Loudness','Mode','Popularity','Speechiness','Tempo','Valence'});
bar_graph = bar(X_names, predictor_importance_est)
title({'Out-of-Bag Permuted Predictor Importance Estimates';'for Attributes (X)'});
ylabel('Classification Error');

%% Plotting 3-dimensional graph for objective function 
hyperparameter_trace = rf_Mdl.HyperparameterOptimizationResults.XTrace;
optimisation_results = rf_Mdl.HyperparameterOptimizationResults;

x = hyperparameter_trace.MinLeafSize;
y = hyperparameter_trace.NumVariablesToSample;
z1 = optimisation_results.ObjectiveMinimumTrace;
x_axis = linspace(max(x), min(x), 100);
y_axis = linspace(min(y), max(y), 100);
[X,Y] = meshgrid(x_axis, y_axis);
Z1 = griddata(x, y, z1, X, Y);
figure
surf1 = surfc(X,Y,Z1,'FaceColor',[0.6350 0.0780 0.1840],'FaceAlpha',0.7); hold on
xlabel('Minimum Leaf Size');
ylabel('Number of Variables to Sample');
zlabel('Objective Function');
title('Hyperparameter Tuning vs Objective Function');

%% Saving the variables in .mat file.
save random_forest.mat
save('final_models/rf_Mdl_optimised.mat','rf_Mdl_optimised')

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