# Lab 10: Boosting
########Group Member: Yuchen Hu, Jiangshan Zhang, Jiaxuan Zhao, Xin Zhou.#######################################
################################################################################################################


# INSTRUCTIONS:
# Run the code below, step-by-step, and answer the questions.
# Submit your R code file with your answer filled in below each question.


# _________________________________________________________________
# Part A: Understanding boosting
#
# Simulation example and code for Gradient Boosting with stumps in 1-d.

# true relationship between x and y
f = function(x) { 10*exp(-x^2)*(-2*x + x^2 + 0.5*x^3 - x^4) }

# Simulate data
set.seed(1)
n = 100  # number of data points to simulate
x = sort(rnorm(n,0,2))  # simulate x's and sort them (the code below assumes the x's are sorted!)
y = f(x) + rcauchy(n,0,0.1) + rnorm(n,0,1)  # simulate outcomes y_1,...,y_n
plot(x,y)  # plot simulated data
x_grid = seq(-5,5,0.01)
lines(x_grid,f(x_grid),col=4)  # plot true function

# QUESTION 1:
# Where is the outlier?
#it is around (0.8,15).
# Which aspect of the data generating process is causing the outlier?
#The error term from the cauchy distribution produce the outlier.

# Algorithm settings
B = 100  # number of boosting iterations to use (i.e., number of trees)
absolute_loss = T  # choice of loss (T: absolute loss, F: square loss)
lambda = 1.0  # shrinkage coefficient between 0 and 1 (where 1 is no shrinkage)
score = function(r) { sum((r-mean(r))^2) }  # RSS of a given set of training outcomes when predicting the average

# Initialize
split = rep(0,B)  # split[b] = split point for tree b (each tree is only a stump)
beta = matrix(0,B,2)  # beta[b,m] = value to predict for region m of tree b
yhat = rep(0,n)  # yhat[i] = predicted outcome value for x[i]

# Run gradient boosting for B iterations
for (b in 1:B) {

    # plot data and current predictions
    plot(x,y)  # data
    lines(x,yhat,col=8)  # current predicted outcome values (light gray)
    lines(x_grid,f(x_grid),col=4)  # true function (blue)
    
    # compute pseudo-residuals
    if (absolute_loss) {
        # for absolute loss: L(y,yhat) = abs(y-yhat)
        r = sign(y - yhat)
    } else {
        # for square loss: L(y,yhat) = 0.5*(y-yhat)^2
        r = y - yhat
    }
    
    # plot pseudo-residuals
    for (i in 1:n) lines(c(x[i],x[i]), c(yhat[i],yhat[i]+r[i]), col=2)  # (red)

    # approximate the gradient by fitting a stump to the pseudo-residuals using square loss
    scores = sapply(1:(n-1), function(i) { score(r[1:i]) + score(r[(i+1):n]) })  # compute RSS for each possible split point
    i = which.min(scores)  # find the best split point, i
    split[b] = (x[i]+x[i+1])/2  # record the split point (using the midpoint between x[i] and x[i+1])
    lines(rep(split[b],2),c(min(y),max(y)),col=1,lty=2,lwd=2)  # plot the best split point (black dotted)
    
    # refine the predicted value in each region of stump b by minimizing loss
    if (absolute_loss) {
        beta[b,1] = median((y-yhat)[x <  split[b]])  # the median minimizes absolute loss
        beta[b,2] = median((y-yhat)[x >= split[b]])
    } else {
        beta[b,1] = mean((y-yhat)[x <  split[b]])  # the mean minimizes square loss
        beta[b,2] = mean((y-yhat)[x >= split[b]])
    }
    
    # update the current predictions
    yhat = yhat + lambda*beta[b, (x >= split[b])+1]
    lines(x,yhat,col=3,lwd=2)  # (green)
    
    Sys.sleep(0.1)  # pause to allow plots to refresh
    
    # readline()  # wait for user to press enter to continue
}

# QUESTION 2:
# What is the interpretation of:
# - the blue curve?
#The blue curve is the true value plot for the function.
# - the light gray curve?
#The current prediction curve for the function in this iteration.
# - the green curve?
#The updated prediction curve for the function in this iteration.
# - the vertical red lines?
#The current residuals for each point.
# - the vertical black dotted line?
#The best split point for the tree in this iteration to produce the minimum difference of sum of squares of models.
# QUESTION 3:
# In words, describe how the current prediction function changes at each iteration,
# relative to the split point of the stump added at that iteration.
# The algorithm split the points at each point, then calculate the mean for two parts and calculate the sum of square,
# then pick the split which has the smallest value of sum of square, and use the mean for the prediction.

# QUESTION 4:
# When absolute_loss=F, many iterations of the algorithm are "wasted" on splits
# near to the outlier.  Why does this happen, even though only two splits are 
# needed to fit the outlier perfectly?
#Since the outlier significantly affects the sum of squares when spliting the points, the split point will be always around it. 
#however, since it is the only point, the mean of splited points changes a little for each iteration, thus, it need many
#iterations to adjust the prediction around it.
# QUESTION 5:
# With 99 split points, we could fit all 100 training data points exactly.
# Why doesn't the boosting ensemble overfit and go through all 100 points exactly, 
# even When lambda=1 (no shrinkage)?
#Since each iteration we only use one split, which need balance all the two parts of points, thus it is hard to exactly
#split every point.

# Perform lasso post-processing to select a subset of the B trees
Phi = matrix(0,n,B)
for (b in 1:B) { Phi[,b] = beta[b,(x>=split[b])+1] }  # construct design matrix
library(glmnet)
fit = cv.glmnet(Phi,y,alpha=1)  # fit lasso
plot(fit)
yhat_lasso = predict(fit, s=fit$lambda.min, newx=Phi)
lasso_coef = predict(fit, s=fit$lambda.min, type="coefficients")
sum(lasso_coef != 0)

# Plot lasso+boosting predictions
plot(x,y)
lines(x_grid,f(x_grid),col=4)  # true function (blue)
lines(x,yhat,col=1)  # boosting predictions (black)
lines(x,yhat_lasso,col=3,lwd=2)  # lasso+boosting predictions (green)


# QUESTION 6:
# How many trees does lasso use?
#99 trees
# Near the outlier, does the lasso+boosting prediction 
#     overfit more or less than the boosting prediction?
# The lasso+boosting prediction overfit more.


# QUESTION 7:
# Now, set absolute_loss=T and rerun the code above starting from the line
# with "set.seed(1)".  
# What is different about the red lines?
# The red line represent for the sign of the residuals.
# What happens with the outlier?
#The outlier doesn't affect the splits too much now.
# Why should using absolute loss perform better when there are outliers?
#because the pseudo residuals won't have too much values when calculating the outlier sum of square.
# QUESTION 8:
# When absolute_loss=T:
#      How many trees does lasso use? 
#43
#      Near the outlier, why is the lasso+boosting prediction 
#          substantially different from the boosting prediction?

#since the lasso use the squre loss, it tend to focus on the outlier more as when absolute_loss is False.


# _________________________________________________________________
# 
# Parts B-D are a modified version of the ISL lab in chapter 8.
#
# Read along in the text for Sections 8.3.2, 8.3.3, and 8.3.4 in the ISL book
# for explanations accompanying these parts.

# _________________________________________________________________
# Part B: Regression Trees

install.packages("tree")
library(tree)
library(ISLR)

# Fit a tree to predict median house prices.
library(MASS)
set.seed(1)
train = sample(1:nrow(Boston), nrow(Boston)/2)
tree.boston=tree(medv~.,Boston,subset=train)
summary(tree.boston)
  # lstat = % lower economic status
  # rm = average number of rooms per dwelling
  # dis = weighted mean of distances to five Boston employment centres

# Visualize the tree
plot(tree.boston)
text(tree.boston,pretty=0)

# Do cross-validation and prune
cv.boston=cv.tree(tree.boston)
plot(cv.boston$size,cv.boston$dev,type='b')
prune.boston=prune.tree(tree.boston,best=8)

# Visualize the pruned tree
plot(prune.boston)
text(prune.boston,pretty=0)

# Evaluate test performance
yhat=predict(tree.boston,newdata=Boston[-train,])
boston.test=Boston[-train,"medv"]

# Plot true vs pred
plot(yhat,boston.test)
abline(0,1)

# Test MSE
mean((yhat-boston.test)^2)

# Test RMSE
sqrt(mean((yhat-boston.test)^2))

# QUESTION 9:
# Visually, do the predictions correlate pretty well with the true values?
# No, the predictions seems have very large variance, thus the mse is not good.
# In units of dollars, how close are the single tree predictions in terms of RMSE?
# For each prediction, the mean difference between prediction and true value is 5k dollar.

# _________________________________________________________________
# Part C: Bagging and Random Forests

# Now, let's do bagging.
dim(Boston)
library(randomForest)
set.seed(1)
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,importance=TRUE)
bag.boston
yhat.bag = predict(bag.boston,newdata=Boston[-train,])

# QUESTION 10:
# How did we use the randomForest function to do bagging?
# let mtry be the number of predictors
  
# Plot true vs pred
par(mfrow=c(1,1))
plot(yhat.bag, boston.test)
abline(0,1)
# Test MSE
mean((yhat.bag-boston.test)^2)
# Test RMSE
sqrt(mean((yhat.bag-boston.test)^2))
  
# QUESTION 11:
# In units of dollars, how close are the bagging predictions in terms of RMSE?
# Is it better than the single tree RMSE?
#For each prediction, the mean difference between prediction and true value is 3.68k dollar.
#Yes, it is better than the single tree.


# What if we use fewer trees (only 25 instead of 500)?
bag.boston=randomForest(medv~.,data=Boston,subset=train,mtry=13,ntree=25)
yhat.bag = predict(bag.boston,newdata=Boston[-train,])
mean((yhat.bag-boston.test)^2)
# Test RMSE
sqrt(mean((yhat.bag-boston.test)^2))

# QUESTION 12:
# Does RMSE get better or worse?
# The RMSE get worse.

  
# Now, let's try a random forest.
# mtry=6 says to use a random subset of 6 predictors when choosing each split.
set.seed(1)
rf.boston=randomForest(medv~.,data=Boston,subset=train,mtry=6,importance=TRUE)
yhat.rf = predict(rf.boston,newdata=Boston[-train,])
mean((yhat.rf-boston.test)^2)
# Test RMSE
sqrt(mean((yhat.rf-boston.test)^2))

# QUESTION 13:
# Are the random forest predictions better or worse than bagging?
# The prediction is better than bagging.

  
# What about variable importance?
# Which variables are most heavily used by *this* trained random forest?
#the lstat is the most heavily used by the random forest.

importance(rf.boston)
varImpPlot(rf.boston)
  # %IncMSE is permutation importance using MSE, scaled by the std dev over trees.
  # IncNodePurity is decrease in RSS due to splits on predictor j, averaged over trees.


# _________________________________________________________________
# Part D: Boosting

library(gbm)
set.seed(1)
  # Gradient boosted trees
  # Defaults:
  #   n.trees = 100 (number of stages to use)
  #   interaction.depth = 1 (stumps)
  #   shrinkage = 0.001
  #   bag.fraction = 0.5 (stochastic boosting using 50% of data at each stage)
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4)
gbm.perf(boost.boston)
summary(boost.boston)

# QUESTION 14:
# What are the top three variables in order of importance?
#lstat rm and dis.

# Plot partial dependence of outcome on rm and lstat
par(mfrow=c(1,2))
plot(boost.boston,i="rm")
plot(boost.boston,i="lstat")


# QUESTION 15:
# Run the code below, and compare the RMSE of boosting with each
# combiation of settings.

# Setting 1:
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)
# RMSE
sqrt(mean((yhat.boost-boston.test)^2))

# Setting 2:
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=5000,interaction.depth=4,shrinkage=0.2,verbose=F)
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=5000)
mean((yhat.boost-boston.test)^2)
# RMSE
sqrt(mean((yhat.boost-boston.test)^2))

# Setting 3: Maybe 5000 trees wasn't enough. Try more trees with small shrinkage.
# Also, let's use cross-validation to estimate test performance and choose n.trees.
boost.boston=gbm(medv~.,data=Boston[train,],distribution="gaussian",n.trees=50000,interaction.depth=4,cv.folds=2,verbose=F)
n.trees.cv = gbm.perf(boost.boston,method="cv")
n.trees.cv
yhat.boost=predict(boost.boston,newdata=Boston[-train,],n.trees=n.trees.cv)
# RMSE
sqrt(mean((yhat.boost-boston.test)^2))


  




  
  