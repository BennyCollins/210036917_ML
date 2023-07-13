close all; clear all; clc;

music_data_dirty = readtable('music_dataset.csv');

%% Removing unnecessary index column.
music_data_dirty = music_data_dirty(:,2:end);

%% Cleaning dataset
%{
Iteratively calculating the VIFs, then removing the feature with the 
highest VIF, if it is over the VIF threshold of 2.5. 
The vif_clean function is defined at the bottom of the file.
%}
music_data_clean = vif_clean(music_data_dirty, 2.5); 
 
%{
Our output tells us to remove the columns corresponding to energy and 
loudness. 
This leaves us with the features: acousticness, danceability, duration_ms, 
instrumentalness, liveness, mode, popularity, speechiness, tempo, and 
valence.
%}

%% Saving the dataset with the relevant variables removed as a csv.
writetable(music_data_clean,'music_data_clean.csv')

%% Saving variables, including cleaned dataset, in .mat file
save VIF_feature_selection.mat

%% Defining functions: vif and vif_clean
%{
vif: function for finding Variance Inflation Factor of a set of features, X 
%}
function VIF = vif(X)
    X_corr = corrcoef(X.Variables);
    VIF = diag(inv(X_corr))';
end

%{
vif_clean: function iteratively calculates the VIFs of a dataset and 
remove, from X, the feature with the highest VIF, if it is over a certain 
threshold, vif_limit. This should eliminate any chance of multicollinearity 
between our features.
%}
function dataset_clean = vif_clean(dataset, vif_limit)
    clean = 0;
    while clean == 0
        VIF = vif(dataset(:,1:end-1));
        disp(VIF);
        [M, I] = max(VIF);
        if M > vif_limit
            variable_list = dataset.Properties.VariableNames;
            dataset(:,I) = [];
            disp('Remove column for the variable:');
            disp(variable_list(I));
        else
            clean = 1;
            dataset_clean = dataset;
        end
    end
end