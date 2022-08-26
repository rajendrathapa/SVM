function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% Let us create a list of C's and sigma's
C_s = [0.01 0.03 0.1 0.3 1.0 3.0 10 30];
sigma_s= [0.01 0.03 0.1 0.3 1.0 3.0 10 30];
% Create an array to store the prediction_err for each permutation of C and sigma values 
% The first row stores the prediction_err, second and third row will store C and sigma values
prediction_err = zeros(length(C_s) , length(sigma_s) );

for ind1 = 1: length(C_s)
  for ind2 = 1: length(sigma_s)
    index = (ind1 -1)  * length(C_s)  + ind2; 
    fit_model = svmTrain(X,y, C_s(ind1), @(x1,x2) gaussianKernel(x1, x2, sigma_s(ind2)) );
    predictions = svmPredict(fit_model, Xval);
    prediction_err(ind1, ind2)  =  mean(double(predictions ~= yval));    
  end
end 
% Sort the prediction_err table 
[min_val, row] = min ( min ( prediction_err, [], 2));
[min_val, col] = min ( min ( prediction_err, [], 1));


% The sorted_pred has lowest prediction_err value and the corresponding C and sigma values on the first row
C = C_s(row);
sigma = sigma_s(col);
% =========================================================================

end
