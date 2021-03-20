### In this part we will do:
### (1) All subset selections with no interaction
### (2) Forward selection with two way interactions
### (3) Backward selection with two way interactions

### Upload the data:

wine = read.csv("white-wine.csv", header=TRUE)

# Install packages for sub set selection
# install.packages("leaps")
library(leaps)

### The regsubsets function is the main function in this package that 
###  allows you to do variable selection
### Setting nvmax = 11, will consider all the possible subsets
### When the summary() on the results is run, it computes the best as the lowest RSS

all.subsets.models = regsubsets(quality ~ .,     data = wine, nvmax = 11)
forward.models     = regsubsets(quality ~ (.)^2, data = wine, nvmax = 66, method = "forward")
backward.models    = regsubsets(quality ~ (.)^2, data = wine, nvmax = 66, method = "backward")

summary.all.subsets.models = summary(all.subsets.models)
summary.all.subsets.models

###
### Among 1 variable models, the model with only alcohol had the lowest RSS.
### Among 2 variable models, the model with alcohol and volatile.acidity has the lowest RSS.
### Among 3 variable models, the model with alcohol and volatile.acidity and residual.sugar has the lowest RSS.
###

### What additional information is contained in the summary?

names(summary.all.subsets.models)

###
### Extract the BIC values.  The lower the BIC, the better the model.
###

bic.all.subsets.models = summary.all.subsets.models$bic
bic.all.subsets.models 

###
### Plot the BIC values versus the number of predictors to see the "best" model:
###

predictors = 1:11
plot(bic.all.subsets.models ~ predictors, ylab="BIC", xlab="Number of Predictors", type="b", pch=19)
abline(v = 8, lty = 2, col = "red")

### Note something interesting: BIC keeps on decreasing but after some point it starts going up; models with 
### more predictors start to perform worse.

### 8 appears to be the lowest one.  Which ones are they?

summary.all.subsets.models

coef(all.subsets.models,8)

### They are: fixed.acidity + volatile.acidity + residual.sugar + free.sulfur.dioxide + density + pH + sulphates + alcohol

### Let's look at the results of the forward and backward too

summary.forward.models  = summary(forward.models)
summary.backward.models = summary(backward.models)

bic.values.forward  = summary.forward.models$bic
bic.values.backward = summary.backward.models$bic

predictors = 1:66
matplot(x = predictors , y = cbind(bic.values.forward, bic.values.backward),type = "l", lty = 1, 
        ylab = "BIC", xlab="Number of Predictors", col = c("blue","red"), lwd = 2 )
abline(v = which.min(bic.values.forward),   lty = 2, col = "blue")
abline(v = which.min(bic.values.backward),  lty = 2, col = "red")
legend("topright", c("Forward","Backward"), lty = 1, col = c("blue","red"), lwd = 2, bty = "n")

### The BIC based best models must be different.

### So what?  What have we done? Which model should we go with?

which.min(bic.values.forward)
which.min(bic.values.backward)

coef(forward.models,  23)
coef(backward.models, 26)

### Some ugly string processing

u1               = toString(names(coef(forward.models, 23))[-1])
u2               = gsub(pattern = ", ",  replacement = " + ", x = toString(u1))
forward.formula  = as.formula(paste("quality ~ ", u2, sep = ""))

v1               = toString(names(coef(backward.models, 26))[-1])
v2               = gsub(pattern = ", ",  replacement = " + ", x = toString(v1))
backward.formula = as.formula(paste("quality ~ ", v2, sep = ""))

### Does it perform better when doing prediction?
### Just for fun let's inculde a model with only alcohol

no.of.folds  = 10
set.seed(666)
index.values = sample(1:no.of.folds, size = dim(wine)[1], replace = TRUE)

mse = matrix(NA, nrow = no.of.folds, ncol = 4)
colnames(mse) = c("Alcohol","All","Forward","Backward")

for (i in 1:no.of.folds)
{
   index.out               = which(index.values == i)                             
   left.out.data           = wine[  index.out, ]                                  
   left.in.data            = wine[ -index.out, ]  
    
   alcohol          = lm(quality ~ alcohol, data = left.in.data)                             
   all              = lm(quality ~ fixed.acidity + volatile.acidity + residual.sugar + free.sulfur.dioxide + 
                                       density + pH + sulphates + alcohol, data = left.in.data)     
   forward          = lm(forward.formula,  data = left.in.data) 
   backward         = lm(backward.formula, data = left.in.data)     

   y =  left.out.data[,12]
     
   mse[i,1] = mean((y - predict(alcohol,  newdata = left.out.data))^2) 
   mse[i,2] = mean((y - predict(all,      newdata = left.out.data))^2)
   mse[i,3] = mean((y - predict(forward,  newdata = left.out.data))^2) 
   mse[i,4] = mean((y - predict(backward, newdata = left.out.data))^2)
  
}

### Now let's look at the RMSE values.

head(mse)

sqrt(apply(mse, MARGIN = 2, FUN = mean))
 
############################################################################