# Lab 6: Caret package tutorial
# 
# Author: Jeffrey W. Miller
# Date: March 14, 2019

#Group member: Jiangshan Zhang, Yuchen Hu, Xin Zhou, Jiaxuan Zhao
# INSTRUCTIONS: Run each step below in R and answer the questions.

# Caret is a powerful R package that provides a unified interface for many machine learning algorithms,
# as well as automating many aspects of preprocessing, tuning, training, and testing.
# This lab will introduce you to some of the functions of the caret package.


# __________________________________________________________________________________________________
# Data

# We'll use the Caravan Insurance Data set from the CoIL 2000 Challenge.
# https://www.kaggle.com/uciml/caravan-insurance-challenge/data

library(ISLR)
dim(Caravan)
str(Caravan)
write.csv(Caravan,"Caravan.csv")  # open in Excel so we can look at it more easily
summary(Caravan)

# Info on the variables is at http://www.liacs.nl/~putten/library/cc2000/data.html

# The outcome of interest is the "Purchase" variable (Yes/No),
# which indicates whether a customer bought a caravan insurance policy.
# The goal is to predict Purchase from the other variables.

# Question: What fraction of customers purchased a policy?
#0.05977
# Question: If you always predict "No", roughly what would your error rate be?
#0.05977
# Question: Suppose you want to choose customers to try to sell a policy.
#    Suppose the cost of pursuing a customer who doesn't purchase
#    is less than the cost of not pursuing a customer who would have purchased.
#    How can we incorporate this into our classification rule?
#generate a loss function which gives more penalty for predicting 'Yes' to 'No', and less penalty for prediction 'No' to 'Yes'.
set.seed(1)  # set random number generator so that we all get the same results
test = 1:1000  # held-out test set
train = setdiff(1:nrow(Caravan), test)  # training set

# __________________________________________________________________________________________________
# caret basics using glm

install.packages("caret")
install.packages("e1071")
library(caret)

# Without using caret, we can fit logistic regression with the built-in glm function:
fit.base = glm(Purchase~., data=Caravan[train,], family=binomial)
fit.base

# In caret, the "glm" method calls the built-in glm function.
# Using trainControl("none") tells caret to just fit the model once, without tuning.
fit = train(Purchase~., data=Caravan[train,], method="glm", trControl=trainControl("none"))
fit
fit$finalModel

# Look at fit.base and fit$finalModel... do they look the same?
#Yes,they look the same.
# Here are the estimated coefficients of the two fitted models:
cbind(coefficients(fit$finalModel),coefficients(fit.base))

# So, what is the purpose of caret?

# The unified interface allows you to easily call many different learning algorithms
# without writing special code to call each of them. It also automates many other tasks.

# We can call the "predict" function as usual to make predictions.
# Predicted probability of Purchase on each customer in the test set:
glm.probs = predict(fit, Caravan[test,], type="prob")
glm.probs

# Question: For what fraction of test customers is the predicted probability 
#    of "Yes" greater than a threshold of 0.5?  How does this compare to the
#    actual fraction of test customers with Purchase="Yes"?
#0.007, which is less than the actual fraction of test customers with purchase="yes"
# The pROC helps make pretty ROC curve plots.
install.packages("pROC")
library(pROC)
roc_curve = roc(Caravan$Purchase[test], glm.probs$Yes, plot=T, main="glm")  # plot the ROC curve

# This shows the sensivity and specificity for a range of thresholds:
coords(roc_curve, "local maximas", ret=c("threshold", "sensitivity", "specificity"))

# Question: Roughly what are the sensitivity and specificity for a threshold of 0.5?
#    (Note: coords can also compute this exactly --- see the pROC documentation.)
#sensitivity:0, specificity:0.993
# To speed up the computations in the rest of the lab, let's use a subset of variables
# that appear to be potentially useful for predicting the outcome:
pvalues = summary(fit$finalModel)$coefficients[,4]
variables = which(pvalues < 0.2)
variables
C = Caravan[,c(names(variables),"Purchase")]
dim(C)

# Question: How many predictors are in C versus in the full Caravan data set?
#There are 17 predictors in C versus in the full Caravan data set.
# ____________________________
# Plotting with caret
# https://topepo.github.io/caret/visualizations.html

featurePlot(C[,1:9], C$Purchase, "density",
            scales = list(x=list(relation="free"), y=list(relation="free")),
            pch = "|",
            auto.key = list(columns=2))

# Question: The PPERSAUT plot looks roughly like a mixture of two Gaussians
#    for each of "Yes" and "No" separately.  Now look at C$PPERSAUT.
#    Comment on how this plot (and the others) are very misleading!
#    This is a common issue with many density plotting routines that use smoothing.

hist(C$PPERSAUT)  # Histograms provide a more "honest" representation.

# ____________________________
# Preprocessing with caret

# Caret has some nice preprocessing functionality:
?preProcess

# Take a look at some of the options of preProcess, e.g.:
#   center: center the data to zero mean
#   scale: scale the data to unit variance
#   corr: filter out highly correlated predictors
#   nzv: remove non-zero variance ("nzv") predictors

# preProcess returns an object that can then be called with "predict"
# to actually perform the preprocessing transformations.
# The purpose of this is that it enables you to take the preprocessing transformations
# "learned" from the training data and apply them to the test data.

prep = preProcess(C[train,], method = c("nzv", "corr", "center", "scale"))
C.tr = predict(prep, C[train,])
dim(C.tr)
dim(C.tr)
summary(C.tr)

# Question: In C.tr, what is the mean of each predictor?
# The mean of each variables are all 0.
C.te = predict(prep, C[test,])
summary(C.te)

# Question: In C.te, why isn't the mean of each predictor equal to 0?
# because we only preprocess the train data set.
# Question: Why not use all of the data (training+test) to "learn" the preprocessing transformations?
# It will induce the information of test set.
# Note: preProcess can also:
# - impute missing data
# - do PCA and ICA
# - apply various transformations such as Box-Cox, spatial sign, etc.

# __________________________________________________________________________________________________
# caret automatically tunes model settings using train/test splits.

# Without using caret, we can do knn with the "class" library:
library(class)
K = 5  # number of neighbors to use in KNN
j = which(names(C)=="Purchase")  #  index of outcome variable
knn.pred = knn(C[train,-j], C[test,-j], C[train,j], k=K, prob=T)  # predict on test set
    # prob=T says to use the probability version of KNN classifier
knn.pred
roc(C$Purchase[test], attr(knn.pred,"prob"), plot=T, main="KNN")  # plot the ROC curve

# caret provides an interface to the same knn function.
# By default, caret does bootstrap train/test splits to choose model settings.

# The following command uses caret to run knn for a bunch of train/test splits
# to estimate performance for each K in some default range, and choose best K:
fit = train(Purchase~., data=C[train,], method="knn", preProcess=c("center","scale"))
fit

# Using preProcess=c("center","scale") tells caret to learn a centering/scaling transformation
# on each training set and apply it to both the training and testing sets.

# Question: What values of K (# neighbors) did caret try?
#    (Note: The knn function uses lowercase k for the # of neighbors.)
# k=5,7,and 9.
# Question: What was the "Accuracy" for each K?
# 0.91, 0.92, and 0.93.
#    (Note: caret uses the term "Accuracy" for error rate.)
#    How does this compare to the naive method of always predicting "No"?
# The accurarcy is less than the naive method.
probs = predict(fit, C[test,], type="prob")
roc(C$Purchase[test], probs$Yes, plot=T, add=T, col=2)
#   Note: The add=T option tells roc to overlay the curve onto the current plot.

# Question: What is the AUC?
# the AUC is 0.6534.

# ____________________________
# caret makes it easy to use a custom protocol to choose model settings

# caret will train the model for each combination of model setting values in tune.grid.
# In this case, the only setting is k, so it will train for each k in the specified range.
tune.grid <- expand.grid(k = seq(10,100,by=10))

# The following tells caret we want to use 10-fold CV, repeated 3 times, 
# for each combination of model settings.
train.control = trainControl(method="repeatedcv", number=10, repeats=3)

# Now we call caret with train.control and tune.grid to carry out this protocol.
fit = train(Purchase~., data=C[train,], method="knn",
            trControl=train.control, tuneGrid=tune.grid, preProcess=c("center","scale"))
fit

# Question: What value of K did caret choose?
#10,20,30,40,50,60,70,80,90,100
#    Are there any other values of K with the same error rate (Accuracy)?
#yes,30~100 has same accuracy.
#    Why would caret choose this K?
# since this model is simplest knn model.
# Question: Do you think that maximizing Accuracy is a good way to choose K for this dataset?
#No, this lead the KNN predicts all result as no.
probs = predict(fit, C[test,], type="prob")
roc(C$Purchase[test], probs$Yes, plot=T, add=T, col=3)

# Question: What is the AUC?    

# AUC is 0.7077
# ____________________________
# Using AUC to measure performance in caret

# Same grid of K's as before
tune.grid <- expand.grid(k = seq(10,100,by=10))

# twoClassSummary is a function that computes binary classification metrics
train.control = trainControl(method="repeatedcv", number=10, repeats=3, 
                    classProbs=T, summaryFunction=twoClassSummary)

# Now we call caret as before but with metric="ROC", which tells caret
# to use AUC (area under the ROC curve) to choose model settings.
# ROC is one of the outputs of the "twoClassSummary" function.
fit = train(Purchase~., data=C[train,], method="knn", metric="ROC",
            trControl=train.control, tuneGrid=tune.grid, preProcess=c("center","scale"))
fit

# Question: Which value of K does caret choose now?
# K=80
probs = predict(fit, C[test,], type="prob")
roc(C$Purchase[test], probs$Yes, plot=T, add=T, col=4)
legend("bottomright", legend=c("K=5","K=5,7,9","more K's","max AUC"), col=1:4, lwd=2)

# Question: What is the AUC?
# auc is 0.7133.
# Plot summary of caret run
trellis.par.set(caretTheme())
plot(fit) 


# ____________________________
# Using a custom performance metric in caret

# Ideally, we would use a loss function takes into account the cost of different
# types of errors, rather than using AUC or error rate.
# caret makes it easy to do this, as follows.

custom_loss = function(data, lev=NULL, model=NULL) {
  y_true = data$obs
  y_pred = data$pred
  FP = (y_pred=="Yes") & (y_true=="No")
  FN = (y_pred=="No") & (y_true=="Yes")
  loss = mean(FP) + 10*mean(FN)
  return(c(loss=loss))
}

# Question: What is the loss matrix for this custom loss function?
#                     Ture Positive      True Negative
#predicted Positive        0                  1
#Predicted Negative       10                  0

# Now we call caret using our custom loss.
tune.grid <- expand.grid(k = c(1:10))
train.control = trainControl(method="repeatedcv", number=10, repeats=3, 
                             classProbs=T, summaryFunction=custom_loss)
fit = train(Purchase~., data=C[train,], method="knn", metric="loss", maximize=F,
            trControl=train.control, tuneGrid=tune.grid, preProcess=c("center","scale"))
fit

# Question: What value of K does caret choose now?
# K=3.

# ____________________________
# caret makes it easy to run MANY different models!

# Let's use a smaller training set so it doesn't too long:
train = 1001:2000

# models to use
models = c("knn","lda","glm","nb","svmRadial","gbm")
n_models = length(models)
AUC = rep(0,n_models)
names(AUC) = models
for (m in 1:n_models) {
    train.control = trainControl(method="repeatedcv", number=10, repeats=3, 
                               classProbs=T, summaryFunction=twoClassSummary)
    fit = train(Purchase~., data=C[train,], method=models[m], metric="ROC",
                trControl=train.control, preProcess=c("center","scale"))
    probs = predict(fit, C[test,], type="prob")
    roc_curve = roc(C$Purchase[test], probs$Yes)
    plot.roc(roc_curve, add=(m>1), col=m, lwd=2, main="ROC curves")
    legend("bottomright", legend=models, col=1:n_models, lwd=2)
    AUC[m] = roc_curve$auc
}
AUC

# Hundreds of models are available via caret:
#    http://topepo.github.io/caret/available-models.html


# ____________________________

# caret also makes it easy to:
# - use parallel computation
# - train your own custom model
# - do stratified data splits













    
    
    