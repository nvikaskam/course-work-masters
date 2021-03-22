### Simple SVM Example

### Let's go ahead and use the (default data):

train = read.csv(file = "train.csv", header = TRUE )
test  = read.csv(file = "test.csv" , header = TRUE )
sol   = read.csv(file = "sol.csv", header = TRUE )

train$Id = NULL
test$Id  = NULL

y = sol$y

### WARNING: Do not use the full train data!  It will take too much time.
### Even though the underlying svm function uses a library written in C++, it is 
###  still pretty slow.
### We'll use only a small fraction of the train data 3000 data points

n     = 3000
train = train[1:n,]

library(e1071)
library(pROC)

### Setting kernel = "linear" will do the hyperplace classification.
### We'll just use the defaults for the svm function for now except 
###  probability = T will enable use to give predicted probabilities later on.
### Note that svm scales the data. 

system.time({
svm_model = svm(as.factor(y) ~ ., data = train , kernel = "linear", probability = T)
})

### Quickly look at some results:

svm_model
svm_model$index
length(svm_model$index)

### svm_model$index tells us the row number of the support vectors.

### Let's get both the training and test AUC's

train_pred = predict(svm_model, newdata = train, probability = T)
y_train    = attr(train_pred, "probabilities")[,1]
head(y_train)

test_pred  = predict(svm_model, newdata = test, probability = T)
y_test     = attr(test_pred, "probabilities")[,1]
head(y_test)

par(mfrow=c(1,2))
roc(response = train$y, predictor = y_train, plot=T, col = "red")
roc(response = y,       predictor = y_test , plot=T, col = "red")
par(mfrow=c(1,1))

### By the way, something strange happens when you set n = 3000.
### The results get worse!

train = read.csv( "train.csv", header = TRUE )
test  = read.csv( "test.csv",  header = TRUE )
sol   = read.csv( "sol.csv",   header = TRUE )

train$Id = NULL
test$Id  = NULL

y = sol$y

###
### At random, pick 5000 rows of the train data.
### You can change  5000 to say 1000 if your tuning takes too much time.
### We could have just taken the first 5000 rows as well.

set.seed(1)

n     = 5000
ind   = sample(1:nrow(train), size = n, replace = F)
train = train[ind,]

### Do you remember what a very large or very small cost does?
### Do you remember what a very large or very small gamma does?

svm_grid = expand.grid( cost = c(0.1 ,1 ,10 ,100, 1000 ), 
                        gamma = c(0.001, 0.1, 0.5 ,2,10,50) )
dim(svm_grid)
m = dim(svm_grid)[1]
m

svm_auc  = rep(0, m)
svm_auc

no_of_folds = 2
set.seed(2000)
index_values = sample(1:no_of_folds, size = nrow(train), replace = TRUE)


system.time({
  
  for (i in 1:m)
  {
    
    tmp_auc     = rep(0, no_of_folds)
    
    for (j in 1:no_of_folds)
    {
      index_out     = which(index_values == j)                             
      left_out_data = train[  index_out, ]                                
      left_in_data  = train[ -index_out, ]   
      
      ###
      ### The default kernel is radial basis so we do not need to explicitly state it.
      ###
      
      tmp_model     = svm( as.factor(y) ~ ., data = left_in_data, cost = svm_grid$cost[i],
                           gamma = svm_grid$gamma[i], probability = T)  
      
      ###
      ### This next line is a bit awkward but it basically extracts the predicted
      ### probabilities.
      ###
      
      tmp_pred     = attr(predict(tmp_model, newdata = left_out_data, probability = T), 
                          "probabilities")[,1]
      
      tmp_auc[j]   = roc(response = left_out_data$y, predictor = tmp_pred, plot=F)$auc[1]      
      
    }
    
    svm_auc[i]     = mean(tmp_auc)
    
  }
  
})

results     = cbind(svm_grid, svm_auc)
results 
best_result = results[which.max(svm_auc),]
best_result

par( mfrow = c(1,2))
boxplot( svm_auc ~ cost,  data = results, xlab = "Cost", ylab = "AUC")
boxplot( svm_auc ~ gamma, data = results, xlab = "Gamma", ylab = "AUC")
par( mfrow = c(1,1))

# the max value of AUc according to the trend is Cost = 0.1 and Gamma = 0.5

final_model = svm(as.factor(y) ~ ., data = train, cost = best_result$cost, 
                  gamma = best_result$gamma, probability = T)

svm_y = attr(predict(final_model, newdata = test, probability = T),
             "probabilities")[,1]

roc_svm = roc(response = y, predictor = svm_y, plot=T, col = "black")
roc_svm$auc

## The ROC curve result is 0.7596, not bad! 