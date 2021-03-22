###################################################################
###    Compare random forest, gradient boosting & linear models   #
###################################################################

# Using the default data
original_train    = read.csv("train.csv", header = T)
original_train$Id = NULL

head(original_train)

set.seed(321)
ind = sample(1:3, size = nrow(original_train), replace = TRUE)

test_index = which(ind == 1)
train_data = original_train[ -test_index, ] 
test_data  = original_train[  test_index, ] 

library(randomForest)
library(gbm)
library(pROC)

set.seed(770077)

system.time({
  rf_model  = randomForest(as.factor(y) ~ . ,  nodesize = 250, data = train_data)
})

y_for_gbm = ifelse(train_data$y == "y", 1, 0)
system.time({
  gb_model  = gbm(y_for_gbm ~ ., data = train_data[,-1], distribution = "bernoulli", cv.folds = 2, 
                  interaction.depth = 6, n.tree = 2000, shrinkage = 0.05 )
})

rf_model
gb_model

optimal_gbm_tree = gbm.perf(gb_model, method = "cv")
optimal_gbm_tree

rf_y = predict(rf_model, newdata = train_data, type = "prob")[,2]
gb_y = predict(gb_model, newdata = train_data, n.tree = optimal_gbm_tree, type="response")

roc_rf = roc(response = train_data$y, predictor = rf_y , plot=T, col = "red")
roc_rf$auc
roc_gb = roc(response = train_data$y, predictor = gb_y, plot=T, add = T, col = "black")
roc_gb$auc
legend("bottomright", c("RF","GBM"), lwd = "2", col = c("red","black"), bty = "n")

rf_y_test = predict(rf_model, newdata = test_data, type = "prob")[,2]
gb_y_test = predict(gb_model, newdata = test_data, n.tree = optimal_gbm_tree, type="response")

roc_rf_test = roc(response = test_data$y, predictor = rf_y_test , plot=T, col = "red")
roc_rf_test$auc
roc_gb_test = roc(response = test_data$y, predictor = gb_y_test, plot=T, add = T, col = "black")
roc_gb_test$auc
legend("bottomright", c("RF","GBM"), lwd = "2", col = c("red","black"), bty = "n")

## Looks like the gbm model won with a higher AUC

###############################################################################
###   Lets try grid search to tune the model 
###############################################################################

original_train    = read.csv("train.csv", header = T)
original_train$Id = NULL

original_train    = original_train[1:3000,]

set.seed(321)
ind = sample(1:3, size = nrow(original_train), replace = TRUE)

test_index = which(ind == 1)
train_data = original_train[ -test_index, ] 
test_data  = original_train[  test_index, ] 

library(gbm)
library(pROC)

train_data$y = ifelse(train_data$y == "y", 1, 0)
y_test       = test_data$y
test_data$y  = NULL

gbm_grid = expand.grid(interaction.depth = seq(from = 6, to = 8, by = 1), 
                       n.trees = seq(from = 200, to = 2000, by = 400), 
                       shrinkage = c(0.001, 0.05, 1), bag.fraction = c(0.5,1))
gbm_grid

m = dim(gbm_grid)[1]
m

gbm_auc  = rep(0, m)
gbm_auc

no_of_folds = 2
set.seed(2000)
index_values = sample(1:no_of_folds, size = dim(train_data)[1], replace = TRUE)

system.time({
  
  for (i in 1:m)
  {
    
    tmp_auc     = rep(0, no_of_folds)
    
    for (j in 1:no_of_folds)
    {
      index_out     = which(index_values == j)                             
      left_out_data = train_data[  index_out, ]                                
      left_in_data  = train_data[ -index_out, ]   
      
      tmp_model     = gbm( y ~ ., data = left_in_data, dist = "bernoulli",
                           interaction.depth = gbm_grid$interaction.depth[i], 
                           shrinkage         = gbm_grid$shrinkage[i], 
                           n.trees           = gbm_grid$n.trees[i],
                           bag.fraction      = gbm_grid$bag.fraction[i])     
      
      tmp_pred     = predict(tmp_model, newdata = left_out_data, type="response", 
                             n.trees = gbm_grid$n.trees[i])   
      
      tmp_auc[j]   = roc(response = left_out_data$y, predictor = tmp_pred ,
                         plot=F)$auc[1]      
      
    }
    
    gbm_auc[i]     = mean(tmp_auc)
    
  }
  
})

results     = cbind(gbm_grid, gbm_auc)
results 
best_result = results[which.max(gbm_auc),]
best_result

boxplot( gbm_auc ~ shrinkage, data = results)

final_model = gbm(y ~ ., data = train_data, dist = "bernoulli", 
                  interaction.depth = best_result$interaction.depth, 
                  shrinkage         = best_result$shrinkage, 
                  bag.fraction      = best_result$bag.fraction, 
                  n.trees           = best_result$n.trees) 

gb_y = predict(final_model, newdata = test_data, 
               n.tree = best_result$n.trees, type="response")

roc_gb = roc(response = y_test, predictor = gb_y, plot=T, col = "black")
roc_gb$auc