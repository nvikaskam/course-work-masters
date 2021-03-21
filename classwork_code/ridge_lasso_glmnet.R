### The most prominent package to perform ridge, lasso, and elastic net is glmnet.
### This package is very fast and reliable and is written by the creators of lasso
###  and elastic net.

# install.packages("glmnet")
library(glmnet)

set.seed(1000)

### We are going to have 50 predictors of which the last z have zero for coefficients 
###  (so they are not really predictor) and n rows of data.  However, we will use about 
###  m = n/2 of the data for assessing prediction accuracy.  A better way is to cross validate but
###  putting a separate test data aside is more convenient for demo purposes

p = 50
n = 150
m = floor(n/2)
z = 20

### Sigma is the amount of "noise" in data.

sigma = 1

### X is our data matrix.  Data are generated from a Gaussian
###  distribution with mean 0 and sd = sigma

x = matrix( rnorm(p*n, sd = sigma), nrow = n, ncol = p)

### I just generate the coefficients at random from uniform(0,1).
### Note z of them will be set to zero.

b = c(runif(p-z), rep(0,z))

### Here are our y's. The +1 adds an intercept of 1.

y = x %*% b + rnorm(n) + 1

### Let's just look at y versus first predictor and last one:

par(mfrow=c(1,2), pch = 19)
plot(y ~ x[,1])
plot(y ~ x[,p])
par(mfrow=c(1,1))

### Split the data into train and test

train_index = 1:m
train   = x[ train_index,]
test    = x[-train_index,]
y_train = y[train_index]
y_test  = y[-train_index]

###
### Next, we create three models: ridge, lasso, and ols.
### For now I will just use the defaults.
### cv.glmnet automatically performs cross validation to 
###  to pick the lamda.  It does the scaling automatically too.

### Extracting the coef from the ridge and lasso requires specifying
###  the value of lambda at which the lowest cv happened.

ridge_model = cv.glmnet(x = train, y = y_train, alpha = 0)
ridge_model$lambda.min
ridge_coef  = coef(ridge_model, s = ridge_model$lambda.min)

lasso_model = cv.glmnet(x = train, y = y_train, alpha = 1)
lasso_model$lambda.min
lasso_coef  = coef(lasso_model, s = lasso_model$lambda.min)

ols_model   = lm(y_train ~ train)
ols_coef    = coef(ols_model)

### Let's look at the results:

results = as.matrix(cbind(c(1,b), ols_coef, as.matrix(ridge_coef), as.matrix(lasso_coef)))
colnames(results) = c("true","ols", "ridge", "lasso")

head(round(results,4))
tail(round(results,4))

summary(results)

boxplot(results, pch = 19, cex.lab = 1.5, cex.axis = 1.5, col = "grey" )

### Predict on the test data next.
### By the way, you could use predict for lm but it expects a data frame and not a matrix.

rmse = function(a,b) { sqrt(mean((a-b)^2)) }

ols_rmse   = rmse(y_test , cbind(1,test) %*% ols_coef )
ridge_rmse = rmse(y_test , predict(ridge_model, newx = test, s = ridge_model$lambda.min))
lasso_rmse = rmse(y_test , predict(lasso_model, newx = test, s = lasso_model$lambda.min))

ols_rmse 
ridge_rmse
lasso_rmse

### What I did favored lasso