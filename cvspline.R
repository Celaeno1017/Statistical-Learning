knot_max=6
nfolds = 10
n=100
# number of folds to use for cross-validation
permutation = sample(1:n)  # random ordering of all the available data
MSE_fold = matrix(0,nfolds,knot_max)  # vector to hold MSE for each fold and each K
for (j in 1:nfolds) {
  pseudotest = permutation[floor((j-1)*n/nfolds+1) : floor(j*n/nfolds)]  # pseudo-test set for this fold
  pseudotrain = setdiff(1:n, pseudotest)  # pseudo-training set for this fold
  for (K in 1:knot_max) {
    model=lm(y[pseudotrain]~bs(x[pseudotrain],df=3+K,degree=3))
    y_hat = sapply(x[pseudotest], function(x0) { predict(model,data=x0) })  # run KNN at each x in the pseudo-test set
    MSE_fold[j,K] = mean((y[pseudotest] - y_hat)^2)  # compute MSE on the pseudo-test set
  }
}
MSE_cv = colMeans(MSE_fold)  # average across folds to obtain CV estimate of test MSE for each K

plot(1:knot_max, MSE_cv)  # plot CV estimate of test MSE for each K

# Choose the value of K that minimizes estimated test MSE
K_cv = which.min(MSE_cv)
K_cv

fit1 = lm(y~bs(x,df=3+K_cv,degree=3))
## df=4 means we want 3 knots (by default, they are 25, 50 and 75 percentiles, but 
## essentially, it means we have 4 slopes to estimate), and degree=1 means we want 
## it linear.
plot(x,y)
lines(x,fitted(fit1),col="red")