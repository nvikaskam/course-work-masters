### This script is for testing a couple of different classifiers
### Tried out the following with ROC Curves:
### 1. Logistic
### 2. Naive Bayes
### 3. Lasso 
### 4. Ridge

original_train = read.csv("train.csv", header = T)

# I remove Id variable since we won't use it.

original_train$Id = NULL

summary(original_train)
str(original_train)

boxplot(log(x1+0.01) ~ y, data = original_train, xlab = "Default y/n", ylab = "Credit Balance Ratio")

set.seed(321)
ind = sample(1:3, size = nrow(original_train), replace = TRUE)

test_index = which(ind == 1)
train_data = original_train[ -test_index, ] 
test_data  = original_train[  test_index, ] 

dim(train_data)
dim(test_data)
  
library(glmnet)
library(e1071)
library(pROC)

nb_model       = naiveBayes(y ~ ., data = train_data)
logistic_model = glm(as.factor(y) ~ ., data = train_data, family = binomial)

set.seed(10^6)
y_numeric    = ifelse(train_data$y == "y", 1, 0)
lasso_model  = cv.glmnet( x = as.matrix(train_data[,-1]), y = y_numeric, alpha = 1, type.measure = "auc", family = "binomial")
ridge_model  = cv.glmnet( x = as.matrix(train_data[,-1]), y = y_numeric, alpha = 0, type.measure = "auc", family = "binomial")


ridge_coef  = coef(ridge_model, s = ridge_model$lambda.min)
lasso_coef  = coef(lasso_model, s = lasso_model$lambda.min) 
logit_coef  = coef(logistic_model)

results           = as.matrix(cbind(logit_coef, ridge_coef, lasso_coef))
colnames(results) = c("logit", "ridge", "lasso")

round(results,5)

nb_y       = predict(nb_model,       newdata = train_data, type = "raw")[,2]
logistic_y = predict(logistic_model, newdata = train_data, type = "response")
lasso_y    = predict(lasso_model,    newx = as.matrix(train_data[,-1]), s = lasso_model$lambda.min, type = "response")
ridge_y    = predict(ridge_model,    newx = as.matrix(train_data[,-1]), s = ridge_model$lambda.min, type = "response")

roc_nb = roc(response = train_data$y, predictor = nb_y, plot=T, col = "green")
roc_nb$auc
roc_logistic = roc(response = train_data$y, predictor = logistic_y, plot=T, add = T, col = "black")
roc_logistic$auc
roc_lasso = roc(response = train_data$y, predictor = as.numeric(lasso_y), plot=T, add = T, col = "red")
roc_lasso$auc
roc_ridge = roc(response = train_data$y, predictor = as.numeric(ridge_y), plot=T, add = T, col = "blue")
roc_ridge$auc
legend("bottomright", c("NB","Logistic","Lasso","Ridge"), lwd = "2", col = c("green","black","red","blue"), bty = "n")


#Next part 
nb_y_test       = predict(nb_model, newdata = test_data, type = "raw")[,2]
logistic_y_test = predict(logistic_model, newdata = test_data, type = "response")
lasso_y_test    = predict(lasso_model, newx = as.matrix(test_data[,-1]), s = lasso_model$lambda.min, type = "response")
ridge_y_test    = predict(ridge_model, newx = as.matrix(test_data[,-1]), s = ridge_model$lambda.min, type = "response")

roc_nb_test = roc(response = test_data$y, predictor = nb_y_test, plot=T, col = "green")
roc_nb_test$auc
roc_logistic_test = roc(response = test_data$y, predictor = logistic_y_test, plot=T, add = T, col = "black")
roc_logistic_test$auc
roc_lasso_test = roc(response = test_data$y, predictor = as.numeric(lasso_y_test), plot=T, add = T, col = "red")
roc_lasso_test$auc
roc_ridge_test = roc(response = test_data$y, predictor = as.numeric(ridge_y_test), plot=T, add = T, col = "blue")
roc_ridge_test$auc
legend("bottomright", c("NB","Logistic","Lasso","Ridge"), lwd = "2", col = c("green","black","red","blue"), bty = "n")

## FINAL STUFF  

predictions = cbind(nb_y_test, logistic_y_test, as.numeric(lasso_y_test), as.numeric(ridge_y_test))
colnames(predictions) = c("NB", "logistic", "Lasso", "Ridge")

pairs(predictions)
cor(predictions)

round(apply(predictions, 2, summary),5)

head(train_data)