###################################################################################
### Applying a random forest on the Spam data set                                 #
###################################################################################

### Get the library that performs random forests:
install.packages("randomForest")
library(randomForest)

### Carefully look at the options for the function: ntree, mtry, nodesize, importance
?randomForest

### Now fit the random forest model
### Let's also time it
### Make sure you set the same seed or you'll get slightly different results

library(DAAG)

head(spam7)

set.seed(83)

system.time({

random.forest.model = randomForest(yesno ~ ., data = spam7, importance =TRUE)

})

random.forest.model

### What would be our predicted value for a new observation we used for the single tree?
### crl.tot (total length of words in capitals) = 10
### dollar (number of occurrences of the \$ symbol) = 0
### bang (number of occurrences of the ! symbol) = 3
### money (number of occurrences of the word ?money?) = 1
### n000 (number of occurrences of the string ?000?) = 0
### make (number of occurrences of the word ?make?) = 1

new.observation.spam = list( crl.tot = 10, dollar = 0, bang = 3, money = 1, n000 = 0, make = 1)

predict(random.forest.model, newdata = new.observation.spam)

# RF predicts spam
# Here's how you can visualize variable importance

varImpPlot(random.forest.model, type = 1, pch = 19)

### Let's demo the variable importance measure using multiple regression
### This is just a function that gets the rmse:

f = function(a,b) sqrt(mean((a - b)^2))

### Suppose we have 5 predictors but actually only 4 of real predictors.
### The 5th predictor gets a coefficient of zero.

set.seed(1)

n = 1500
p = 5
X = matrix( rnorm( n * p ) , n , p)
colnames(X) = paste("x", seq(1,p), sep = "")

dim(X)
head(X)

### Here are the coefficient.  Note the last one is zero.
B = c(round(10 * rnorm(p-1)),0)

### Here's our full data:
y   = X %*% B + rnorm(n)
dat = data.frame(y, X)
head(dat)

### Separate 2/3 of data to train, 1/3 to test:
train  = dat[1:1000,   ]
test   = dat[1001:n, -1]
y_test = y[1001:n]

### Let's see multiple regression can pick this up.
### Recall -1 in the model says don't include the intercept

g = lm(y ~ . -1, data=train)

summary(g)

g_rmse = f( y_test, predict(g, newdata = test))
g_rmse

### Let's go through each variable in the test data
### permute (shuffle) each variable and "drop" into the model:
### sample(x) just does a random permutation on the data

rmse  = numeric(5)

for (i in 1:p)
{
  tmp_test     = test
  tmp_test[,i] = sample(tmp_test[,i])
  rmse[i]      = f(y_test, predict(g, newdata = tmp_test))
}

### Let's look at the results:
results = data.frame(Var_left_out = c("None",colnames(X)), test_rmse = round(c(g_rmse, rmse),3))
results