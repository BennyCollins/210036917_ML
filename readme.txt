[Title](210036917.pdf)

Dependencies:
Python:
pandas
numpy
matplotlib.pyplot
seaborn
statsmodels
sklearn

MATLAB: 
Machine Learning Statistics Toolbox



Dataset Files:
Raw data: music_genre.csv
Dataset after EDA: music_dataset.csv
VIF-filtered dataset: music_dataset_VIF.csv

MATLAB Files:
VIF_feature_selection.m: This file contains a function that calculates VIFs for each variable in the dataset and then filters out the attribute with the highest VIF that is over a specified threshold.

dataset_partitions.m: The files carries out the train/test split and 10-fold cross-validation partition of our data, saving the partitions and test and train sets for the attributes, X, and target variable, Y.
dataset_partitions.mat: Saved split and partitions for dataset that hasn't been filtered based on VIF.
dataset_partitions_VIF.mat: Saved split and partitions for dataset that's been filtered based on VIF, with a threshold of 2.5.
dataset_partitions_VIF5.mat: Saved split and partitions for dataset that's been filtered based on VIF, with a threshold of 5.

multiple_multinomial_logistic_regression.m: The file imports train/set sets and 10-fold CV partitions from dataset_partitions.mat (this can be adjusted to dataset_partitions_VIF.mat or dataset_partitions_VIF5.mat). It then calculates classification error, precision, recall, F1-score and confusion matrices for train/test sets. The 10-fold CV error is also calculated, and a final model is fit on the training set.
multiple_multinomial_logistic_regression_VIF.mat: Variables saved for dataset that's been filtered based on VIF, with a threshold of 2.5.
multiple_multinomial_logistic_regression_VIF5.mat: Variables saved for dataset that's been filtered based on VIF, with a threshold of 5.

random_forest.m: The file imports train/set sets and 10-fold CV partitions from dataset_partitions.mat. Hyperparameters are set to be optimised. It then fits two models using the optimised hyperparameters, and calculates classification error, precision, recall and F1-score for train/test sets. A confusion matrix is produced for the test set. The 10-fold CV error is also calculated, and a final model is fit on the training set. On top of this, there are a number of visualisations of hyperparameter tuning.

EDA_graphs: A folder of graphs (distributions, correlation matrix) acquired in EDA.
matlab_graphs: A folder of graphs, tables and matrices acquired during model fitting.


How to run tests on final models with test sets:
final_models: A folder containing the final dataset train/test sets as well as the final models.
final_model_partitions.mat: Saved final dataset train (X_train, Y_train) and test (X_test, Y_test) sets for attributes, X, and target variable, Y.
mnr_Mdl.mat: Saved final model for multiple multinomial logistic regression.
rf_Mdl_optimised.mat: Saved final model for random forest.
multinomial_logistic_regression_final_model.m: File imports the train/test sets from final_model_partitions.mat and the multinomial logistic regression model from mnr_Mdl.mat. The the file then calculates test error, training error, micro and macro-average precision, recall and F1 score. The file, finally, produces the test set's confusion matrix for the model.
random_forest_final_model.m: File imports the train/test sets from final_model_partitions.mat and the random forest model from rf_Mdl_optimised.mat. The the file then calculates test error, training error, micro and macro-average precision, recall and F1 score. The file then calculates the out-of-bag permuted predictor importance estimates, and producing the test set's confusion matrix for the model.
