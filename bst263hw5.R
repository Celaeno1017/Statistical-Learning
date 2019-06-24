####1(a)####
set.seed(1)  # set random number generator
n = 1000  # number of samples
x = 5*runif(n)  # simulate training x's uniformly on the interval [0,5]
sigma = 0.3  # standard deviation of the noise
f = function(x) { cos(x) }  # f(x) = true mean of x given y
y = f(x) + sigma*rnorm(n)  # simulate training y's by adding N(0, sigma^2) noise to f(x)
plot(x,y,col=2,pch=20,cex=2)  # plot training data

KNN = function(x0, x, y, K) {
  distances = abs(x - x0)  # Euclidean distance between x0 and each x_i
  o = order(distances)  # order of the training points by distance from x0 (nearest to farthest)
  y0_hat = mean(y[o[1:K]])  # take average of the y values of the K nearest training points
  return(y0_hat)  # return predicted value of y
}

####1(b)####
MSE=matrix(0,nrow = 20,ncol=25)
for (r in 1:25) {
  i = 20  
  x_train = 5*runif(i)  
  sigma = 0.3  
  f = function(x) { cos(x) }  
  y_train = f(x_train) + sigma*rnorm(i)  
  
  for (n in 1:20) {
    y_pred<-sapply(x, function(x0) { KNN(x0, x_train[1:n], y_train[1:n], 1) })
    MSE[n,r] = mean((y_pred - y)^2)
  }
  
}

####1(c)####
MSE_N=rowMeans(MSE)
plot(MSE_N,xlab = 'n',ylab='MSE(n)',main='The change of MSE vs n')


####2(a)####
n_test = 100000
x_test = 5*runif(n_test)  # simulate test x's from true data generating process
y_test = f(x_test) + sigma*rnorm(n_test)  # simulate test y's from true data generating pro

# Simulate dataset
set.seed(50)  # set random number generator
n = 30  # number of samples
x = 5*runif(n)  # simulate training x's uniformly on the interval [0,5]
sigma = 0.3  # standard deviation of the noise
y = f(x) + sigma*rnorm(n)  # simulate training y's by adding N(0, sigma^2) noise to f(x)

# Compute "ground truth" estimate of test performance, given this training set
K = 1  # number of neighbors to use in KNN
y_test_hat = sapply(x_test, function(x0) { KNN(x0, x, y, K) })  # run KNN at each x in the test set
MSE_test = mean((y_test - y_test_hat)^2)  # compute MSE on test set

# Repeatedly run CV for a range of nfolds values
nfolds_max = n  # maximum value of nfolds to use for CV
nreps = 1000  # number of times to repeat the simulation
MSE_cv = matrix(0,nreps,nfolds_max)  # vector to hold CV estimate of MSE for each rep and each fold
for (r in 1:nreps) {  # run the simulation many times
  for (nfolds in 1:nfolds_max) {
    permutation = sample(1:n)  # random ordering of all the available data
    MSE_fold = rep(0,nfolds)  # vector to hold MSE for each fold and each K
    for (j in 1:nfolds) {
      pseudotest = permutation[floor((j-1)*n/nfolds+1) : floor(j*n/nfolds)]  # pseudo-test set
      pseudotrain = setdiff(1:n, pseudotest)  # pseudo-training set
      y_hat = sapply(x[pseudotest], function(x0) { KNN(x0, x[pseudotrain], y[pseudotrain], K) })  # run KNN at each x in the pseudo-test set
      MSE_fold[j] = mean((y[pseudotest] - y_hat)^2)  # compute MSE on the pseudo-test set
    }
    MSE_cv[r,nfolds] = mean(MSE_fold)
  }
}

# Compute the MSE, bias, and variance of the CV estimate of test MSE, for each value of nfolds
mse = colMeans((MSE_cv - MSE_test)^2)
bias = colMeans(MSE_cv) - MSE_test
variance = apply(MSE_cv,2,var)

plot(1:nfolds_max, type="n", ylim=c(0,max(variance[2:nfolds_max])*1.1), xlab="nfolds", ylab="variance", main="Variance of the CV estimates with n=18")
lines(1:nfolds_max, variance, col=4, lwd=2)
legend("topright", legend=c("variance"), col=c(4), lwd=2)

