###Digit Recognition Assignment 5

train_rows = 5000
test_rows  = 2500

setwd('/Users/trestor/Downloads/Full Data for Assignment 5')

original_train = read.csv("train.csv", header = F, nrows = train_rows)
original_test  = read.csv("test.csv", header = F, nrows = test_rows)


head(original_train)

dim(original_train)
dim(original_test)

dim(test)

train$V785 = as.factor(train$V785)
test$V785  = as.factor(test$V785)

plot(table(train$V785))
summary(train$V785)/train_rows
summary(test$V785)/test_rows

###
### The last column ncol(train) is excluded since this is now a
### categorical variable.
###

plot(sort(apply(train[,-ncol(train)], 2, sd)))


show_an_image = function(n)
{
  v  = as.numeric(train[n,1:784])
  im = matrix(v,28,28)
  im = im[,nrow(im):1]
  image(im, col = gray((0:255)/255), main = train[n,785])
}

###
### Let's look at the pictures of images:
###  1, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
### You can try others too.
###
### The image numbers must be between 1 and train_rows
###

par(mfrow = c(3,4))
for (i in c(1, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500))
{
  show_an_image(i)
}
par(mfrow = c(1,1))

###
### 20 was sort of a guess!
###

cut_off     = 1
bad_columns = which(apply(train[,-ncol(train)], 2, sd) < cut_off)
length(bad_columns)

train = train[,-bad_columns]
test  = test[,-bad_columns]
ncol(train)
library(e1071)

svm_model = svm(V785 ~ ., data = train)
?svm
svm_y     = predict(svm_model, newdata = test)

confusion_matrix = table(predicted = svm_y, actual = test$V785)
confusion_matrix

sum(diag(confusion_matrix))
nrow(test)
diag(confusion_matrix)/table(test$V785)
error_rate = 1 - ( sum(diag(confusion_matrix)) / nrow(test))
1-error_rate
#92.04% Accuracy 
###
### cross = 3 means do a 3 fold cross validation.  This is to save some time.
###


set.seed(1010)
system.time({
  tune_svm = tune(svm , V785 ~ ., data = train, kernel ="radial", 
                  ranges =list(cost = c(1 ,5,10  ), gamma = c( 0.001, 0.01, 1 )),
                  tunecontrol = tune.control(cross = 3) )
})

tune_svm 

###
### summary(tune_svm) will display all the results.
###

svm_final   = svm(V785 ~ ., data = train, cost = tune_svm$best.parameters$cost,
                  gamma = tune_svm$best.parameters$gamma)
svm_y_final = predict(svm_final, newdata = test)

confusion_matrix_final = table(predicted = svm_y_final, actual = test$V785)
confusion_matrix_final

error_rate_final = 1 - ( sum(diag(confusion_matrix_final)) / nrow(test))
1-error_rate_final

###PCA and NN

# You can write R code here and then click "Run" to run it on our platform

library(readr)

# The competition datafiles are in the directory ../input
# Read competition data files:
train <- original_train
test <- original_test

# Write to the log:
cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

X <- train[,-785]
Y <- train[,785]
trainlabel <- train[,785]
#Reducing Train using PCA
Xreduced <- X/255
Xcov <- cov(Xreduced)
pcaX <- prcomp(Xcov)
# Creating a datatable to store and plot the
# No of Principal Components vs Cumulative Variance Explained
vexplained <- as.data.frame(pcaX$sdev^2/sum(pcaX$sdev^2))
vexplained <- cbind(c(1:784),vexplained,cumsum(vexplained[,1]))
colnames(vexplained) <- c("No_of_Principal_Components","Individual_Variance_Explained","Cumulative_Variance_Explained")
#Plotting the curve using the datatable obtained
plot(vexplained$No_of_Principal_Components,vexplained$Cumulative_Variance_Explained, xlim = c(0,100),type='b',pch=16,xlab = "Principal Componets",ylab = "Cumulative Variance Explained",main = 'Principal Components vs Cumulative Variance Explained')
#Datatable to store the summary of the datatable obtained
vexplainedsummary <- vexplained[seq(0,100,5),]
vexplainedsummary
#Storing the vexplainedsummary datatable in png format for future reference.
library(gridExtra)
png("datatablevaraince explained.png",height = 800,width =1000)
p <-tableGrob(vexplainedsummary)
grid.arrange(p)
dev.off()
Xfinal <- as.matrix(Xreduced) %*% pcaX$rotation[,1:45]

#Making training labels as factors
trainlabel <- as.factor(trainlabel)
library(nnet)
Y <- class.ind(Y)
print(X[1:5,1:5])

#We choose no_of_nodes=150 and maxiter=100 (change it as a trade-off between running time and accuracy)

#Training the nnet on totat_training_set
finalseed <- 150
set.seed(finalseed)
model_final <- nnet(Xfinal,Y,size=150,softmax=TRUE,maxit=130,MaxNWts = 80000)

#Load test to reduced and normalize it for predictions
testlabel <- as.factor(test[,785])

#Applying PCA to test set
testreduced <- test[,-785]/255
testfinal <- as.matrix(testreduced) %*%  pcaX$rotation[,1:45]

#Calculating Final Accuracies
prediction <- predict(model_final,testfinal,type="class")

prediction <- as.data.frame(prediction)

confusion_matrix_final = table(predicted = prediction, actual = original_test$V785)
confusion_matrix_final

error_rate_final = 1 - ( sum(diag(confusion_matrix_final)) / nrow(test))
1-error_rate_final
#95.44

#XGBoost
#https://www.kaggle.com/evelynstamey/digit-recognizer/digits-xgboost-with-96-4-accuracy/code

library(readr)
library(ggplot2)
library(caret)
library(Matrix)
library(xgboost)

# read training and testing datasets
TRAIN <- original_train
TEST <- original_test

# separate multi-level, categorical response variable ("label") from the remaining predictor variables in the training dataset ("TRAIN")
LABEL <- TRAIN[,785]
TRAIN$V785 <- NULL

# find and remove vectors that are linear combinations of other vectors
LINCOMB <- findLinearCombos(TRAIN)
TRAIN <- TRAIN[, -LINCOMB$remove]
TEST <- TEST[, -LINCOMB$remove]

# find and remove vectors with near-zero variance
NZV <- nearZeroVar(TRAIN, saveMetrics = TRUE)
TRAIN <- TRAIN[, -which(NZV[1:nrow(NZV),]$nzv == TRUE)]
TEST <- TEST[, -which(NZV[1:nrow(NZV),]$nzv == TRUE)]

# re-attach response variable ("LABEL") to training dataset ("TRAIN")
TRAIN$V785 <- LABEL
# define xgb.train parameters
PARAM <- list(
  # General Parameters
  booster            = "gbtree",          # default
  silent             = 0,                 # default
  # Booster Parameters
  eta                = 0.05,              # default = 0.30
  gamma              = 0,                 # default
  max_depth          = 5,                 # default = 6
  min_child_weight   = 1,                 # default
  subsample          = 0.70,              # default = 1
  colsample_bytree   = 0.95,              # default = 1
  num_parallel_tree  = 1,                 # default
  lambda             = 0,                 # default
  lambda_bias        = 0,                 # default
  alpha              = 0,                 # default
  # Task Parameters
  objective          = "multi:softmax",   # default = "reg:linear"
  num_class          = 10,                # default = 0
  base_score         = 0.5,               # default
  eval_metric        = "merror"           # default = "rmes"
)

# convert TRAIN dataframe into a design matrix
TRAIN_SMM <- sparse.model.matrix(Xfinal$train.V785 ~ ., data = Xfinal)
TRAIN_XGB <- xgb.DMatrix(data = TRAIN_SMM, label = Xfinal$train.V785)

# set seed
set.seed(1)

# train xgb model
MODEL <- xgb.train(params      = PARAM, 
                   data        = TRAIN_XGB, 
                   nrounds     = 400, # change this to 400
)

# attach a predictions vector to the test dataset

# use the trained xgb model ("MODEL") on the test data ("TEST") to predict the response variable ("LABEL")
TEST_SMM <- sparse.model.matrix(V785 ~ ., data = TEST)
PRED <- predict(MODEL, TEST_SMM)
PRED
confusion_matrix_final = table(predicted = PRED, actual = original_test$V785)
confusion_matrix_final

error_rate_final = 1 - ( sum(diag(confusion_matrix_final)) / nrow(test))
error_rate_final
#nrounds=400, 93.28% accuracy

#Preprocesing - nzvar, lincomb, pca
#Logistic Regression
#Naive Bayes
#XGBoost
#Adaboost
#RF
#TRAIN,TEST - nzvar and lincomb train

#KNN
install.packages('FNN')
library('FNN')
?knn
dim(TEST)
fnn.kd <- knn(Xfinal, testfinal, as.factor(TRAIN$V785) , k=1 , algorithm=c("kd_tree"))
confusion_matrix_final = table(predicted = fnn.kd, actual = TEST$V785)
error_rate_final = 1 - ( sum(diag(confusion_matrix_final)) / nrow(test))
1-error_rate_final
#94.2 error rate
