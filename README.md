# MachineLearningExercises
Linear Regression, Logistic Regression, Naive Bayes, and Clustering Exercises for Springboard Data Intensive Course

This respository shall house the Machine Learning exercises for Springboard Data Science Intensive Course.

LINEAR RERESSION MINI-PROJECT
The first exercise submitted is the one on Linear Regression.  It is a study of the Boston Housing project and comes from the Harvard CS109 course, lab4.  The data set contains housing values in the Boston area along with 13 attributes or variables related to the house such as the location, average number of rooms, crime rate, property tax rate, etc. 

After loading the data into a pandas dataframe, we created scatter plots of the housing price versus many of the variables to look for a correlation between the variable and the housing price.  When the price was plotted versus crime rate, we observed an inverse relationship.  When the price was plotted versus average number of rooms, we observed a positive linear relationship. When price was plotted versus percent lower status of the population, we observed a negative linear relationship.  The plots of housing versus pupil teacher ratio, lots zoned over 25,00- sq ft, and proportion of non-retail businesses showed no correlation.

We created several investigatory plots. First, we used Seaborn's Regplot command to fit a linear regression model to the data for housing price versus average number of rooms. Then we observed the histogram plot of "CRIM", the crime rate per capita. We plotted the data with and without taking the log.  The distribution wtihout taking the log is skewed and not symmetrical.  By taking the log of the data, we were able to create a more symmetrical distribution and we were able to observe the bi-modal nature of the distribution that is not apparent without taking the log.  Then we created several more distribution plots to look for correlations in predictors (the variables). We compared the distriubtions of "RM", average number of rooms, to "PTRATIO", average pupil teacher ratio.  Then we compared "RM" to "LSTAT" and "ZN".  The only two that looked similar were "RM" and "LSTAT". Since visualizing correlations is best done using scatter plots, we went ahead and created some scatter plots of "RM" versus "PTRATIO", "LSTAT", "ZN", and "TAX".  There did appear to be a correlation between "LSTAT" and "RM".

We created linear regression models utilizing two different tools, statsmodels and sklearn. 

First, with statsmodels, we created an ordinary least squares model for price versus all the predictors.  We looked at the summary report.  Then we plotted the predicted prices generated from the model versus the actual prices. We observed a few oddities such as a negative price prediction and a "ceiling" effect at the highest price, and data fanning out at the bottom left side.

Second, we used sklearn to fit a linear model using all of the predictors.  We printed out the intercept and coefficients for the model. We re-created the model specifying that it not fit an intercept term. This would force the model to go through the origin.  For this particular example of housing prices, doing this would not make much sense because there are no houses with a price of zero and there is no real data available near zero and we would expect there to be a positive y intercept.  Next, we plotted the distribution of the predicted prices noting that there are predicted prices below zero and that the distribution is skewed to the right.  Again, we plotted the predicted price versus the actual price and compared this to the plot earlier from the ols model we created with statsmodels.  The model created from sklearn appeared more accurate.  One advantage of using statsmodels was the benefits of more information generated in the summary report. We evaluated the model using Sum of Squares, the coefficient of determination, mean squared error, and F-statistic 
using the given formulas.

Next, we created a new model for "PRICE" versus the one predictor "PTRATIO" using statsmodel.  We compared the hand generated evaluation metrics to the statsmodels generated values for the coefficient of determination and the F-statistic. We discovered that the F-statistics is just the T-statistic squared for simple linear regression.

Finally, we created a model for "PRICE" versus the three predictors "PTRATIO", "CRIM", and "RM".  Then yet another model adding two more predictors "ZN" and "LSTAT".  We compared the two models looking at the differences in the AIC score and computing the change in F-statistic.  We observed that the model with more predictors was an improvement.

Since linear regression is valid assuming certain assumptions, we investigated these assumptions. Using the model above for "PRICE" versus the three predictors, we plotted the residuals versus the fitted values.  Since we observed no pattern, we could assume that the residuals are indeed normally distributed.  We also did a quantiles plot to show normal distribution.  Finally, we created a leverage veresus normalized residuals squared plot and aninfluence plot to identify outliers and high leverage data points.  The identified points were deleted, the model recreated, and the two compared. It was noted that data points that are outliers and high leverage points can unduly affect the model and should be eliminated while points that are outliers that do not affect the model do not need to be eliminated. 


LOGISTIC REGRESSION MINI-PROJECT
The second Machine Learning excercise is on Logistic Regression.  It is from the Harvard CS109 course, Lab5 on classification. The dataset is a set of heights and weights for males and females.  

First we used Seaborn to create a scatter plot of weight versus height for males and females (the "hue" setting is on gender so the data points are colored differently per gender).

We used sklearn to split the data into training and test sets. We fit a logistic regression model on the training data using the default settings.  Then we used the test data to check the accuracy which was 0.9252 or 93%. 

Using the sklearn KFold package and the given function "cv_score", we did  5-fold cross validation for a basic logistic regression model without regularization.  The accuracy score was 0.917.

We fit a logistic regression model for a given set of C parameters and identified the C parameter that gave the highest accuracy score. We chose the "liblinear" solver and "l1" optimization and set the tolerance to 1e-17.  With the penalty score set at "l2", the differences between the accuracy measures for the different C parameters was not large enough to be able to differentiate a clear "winner".  Therefore the penalty setting was changed to "l1".  At this point, it was also observed that when the cell was re-run with the same data, the results were not consistent.  This is described in the user's manual.  For this reason, the tolerance level was changed until consistent results were achieved. The parameter value that obtained the highest accuracy was C= 0.1.

Using the value 0.1 for C, the logistic regression model was fit to the training data and then accuracy was measured on the test set. The result was accuracy of 0.9228.

Next, we used the sklearn GridSearchCV tool to repeat what we had just done by hand.  The result was best C parameter value of 0.1 and accuracy measure of 0.9228.  For comparison purposes, we also used the function "cv_score" to determine the accuracy score and the result was 0.9240.

We read about the mathematical theory of logistic regression and implemented the given demonstration of performing the five fold cross validation for optimizing parameters.  The results were C=0.01 and train accuracy = 0.92 and test accuracy = 0.92.  A plot of the classifier was implemented showing the decision boundary.

We learned how to obtain the highest probability classification by using maximum likelihood estimation.  We saw how sklearn can compute an array of probabilities for our samples.

Finally, we learned about descriminative versus generative classifiers.  The descriminative classifier finds a soft boundary between classes while a generative classifier finds the distribution of each class.  We saw a plot of the probabilities output by sklearn that shows several decision boundaries with their associated probabilities.

NAIVE BAYES MINI PROJECT

The Dataset used for this exercise is a subset of movie reviews from the Rotten Tomatoes database. 

We learned the basics of text analysis.  We converted the text into numerical feature vectors contained in a matrix.  The columns represent the features/words and the rows are the numerical representation of a "document". This is called a "Bag of Words" representation. 

We implemented a Naive Bayes Classifier and then used K-fold cross validation to optimize the parameters for the classifier. We plotted the cumulative Distribution of the Document Frequencies to help us choose the min_df for feature extraction.  When building the vocabulary (the features), terms that have a document frequency below min_df (can be proportion of documents of integer input) will be ignored.  We saw that the steep part of the curve occured for min_df < 1.

We learned how to identify words that have a high probability of signifying a postive review and words that have a low probabiity of signifying a positive review (Feature Selection). 

We saw an example of mis-classification illustrating some challenges of text interpretation and sentiment classification.  

Finally, we tested several other classifiers such as Random Forest and LinearSVC and compared the results. We also tried using n-grams instead of words, td-idf weighting, and setting max_df.  The optimized Naive Bayes classifier produced the best results for a model that generalized well to new data.  The LinearSVC model had a little higher accuracy but given the difference in the training accuracy to the test accuracy, we saw that the model did not generalize well to new data.



