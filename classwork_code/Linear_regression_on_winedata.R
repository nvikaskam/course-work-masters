
##########################################################################################
### R Scripts to try some linear models with and without interactions on the wine data set
##########################################################################################

### Upload the data:
wine = read.csv(file = "red-wine.csv", header=TRUE)

summary(wine) 
head(wine) 

# Let's look at the histogram of the Y varibale.

hist(wine$quality, col="grey")

for (i in 1:11) 
  plot(quality ~ wine[,i], xlab=colnames(wine)[i], data = wine)

dev.off()

### We probably don't need any transformations

### Let's fit a large model and see the output of the result

initial.large.model = lm(quality ~ .,data = wine)

initial.large.model

# Lets check the summary of the multiple reg model

summary(initial.large.model)

### Lets get a prediction for a new row

new.value = list(
                     fixed.acidity          =	4.00,
                     volatile.acidity     	=	1.00,
                     citric.acid          	=	0.50,
                     residual.sugar       	=	5.00,
                     chlorides          	=	0.05,
                     free.sulfur.dioxide   	=	25.00,
                     total.sulfur.dioxide 	=	150.00,
                     density                =	1.00,
                     pH                   	=	3.00,
                     sulphates             	=	0.75,
                     alcohol   	         =	11.00)

### Fetch the prediction

predict(initial.large.model, newdata = new.value)

### It is pretty low quality!

### "fitted.values" contains all our predicted values (the y-hats)

predicted.values = initial.large.model$fitted.values
head(wine)
observed.value = wine[,12]

### Let's just look at a few lines of observed vs. predicted values.

head(cbind(observed.value, predicted.values))

plot(observed.value, predicted.values, pch=19)

### Finally, lets get the training RMSE value.
training.rmse = sqrt(mean((observed.value - predicted.values)^2))
training.rmse

### When predicting, we are going to be off by about 0.75 points.

### Let's try a 10-fold cross validation

### First let's see how we get the dimensions of our data.

dim(wine)
dim(wine)[1]

### At random assign values 1 to 10 to each row.  10 comes from our "10"-fold cv.

no.of.folds = 10
no.of.folds 

### At random pick each row to be in one of the 10 folds. Note that the fold may not 
### contain equal number of rows.  This is okay as long as we have a decent amount of data.
###
### The function set.seed() sets the random number generator to a starting value.
### This will ensure that we will all get the same results.

set.seed(778899)

index.values = sample(1:no.of.folds, size = dim(wine)[1], replace = TRUE)
head(index.values)
table(index.values)/dim(wine)[1]

### The vector test.mse is going to contain the k mse values.

test.mse = rep(0, no.of.folds)
test.mse

for (i in 1:no.of.folds)
{
   index.out            = which(index.values == i)                             ### These are the indices of the rows that will be left out.
   left.out.data        = wine[  index.out, ]                                  ### This subset of the data is left out. (about 1/10)
   left.in.data         = wine[ -index.out, ]                                  ### This subset of the data is used to get our regression model. (about 9/10)
   tmp.lm               = lm(quality ~ ., data = left.in.data)                 ### Perform regression using the data that is left in.
   tmp.predicted.values = predict(tmp.lm, newdata = left.out.data)             ### Predict the y values for the data that was left out
   test.mse[i]          = mean((left.out.data[,12] - tmp.predicted.values)^2)  ### Get one of the test.mse's
}

test.rmse = sqrt(mean(test.mse))
test.rmse 
training.rmse

### Note that the training rmse is slightly lower

###################################################################################
### Lets try and build some models with interactions
###################################################################################

### The notation (.)^3 tells are to create all 3 way interaction terms.

model.with.interaction = lm(quality ~ (.)^3, data = wine)

summary(model.with.interaction)

summary(initial.large.model)

### Let's try a simple prediction

predict(model.with.interaction, newdata = new.value)
predict(initial.large.model,    newdata = new.value)

### Let's do our cross validation

predicted.values.with.interaction = model.with.interaction$fitted.values

training.rmse.with.interaction    = sqrt(mean((observed.value - predicted.values.with.interaction)^2))
training.rmse.with.interaction
training.rmse

### We should not be surprised that the model with interactions has a smaller training rmse its more complex

### Let's look at the test rmse for the model with interactions

test.mse.with.interaction = rep(0, no.of.folds)

for (i in 1:no.of.folds)
{
   index.out            = which(index.values == i)                             
   left.out.data        = wine[  index.out, ]                                
   left.in.data         = wine[ -index.out, ]                                  
   tmp.lm               = lm(quality ~ (.)^3, data = left.in.data)                
   tmp.predicted.values = predict(tmp.lm, newdata = left.out.data)   
   
   ### Important: left.out.data[,12] has the observed quality values
 
   test.mse.with.interaction[i]          = mean((left.out.data[,12] - tmp.predicted.values)^2)  
}

test.mse.with.interaction
test.rmse.with.interaction = sqrt(mean(test.mse.with.interaction))
test.rmse.with.interaction
test.rmse

### The result with the interactions is worse!  The model with all interaction is overfitting