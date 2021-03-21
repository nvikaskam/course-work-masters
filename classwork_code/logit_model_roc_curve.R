### This is a demo of logistic regression for classifying spam

install.packages("DAAG")
library("DAAG")

### The main function to run logistic regression is glm = generalized linear model
### To run logistic regression, you specifically need to set the "family" input to binomial. 

head(spam7)
summary(spam7)

### Always check the rate of your event of interest.
### Rare events are much harder to model and you may need specialized method.

logistic_model  = glm(yesno ~ . ,  data = spam7, family = binomial)

summary(logistic_model)

logistic_model$converged

### You get a warning message.
### This could be due to :
###  1) Just a warning that some predicted probabilities are too close to 0 or 1 
###     which caused some precision problems.
###  2) The "optimization" process can not find a proper solution.

### Let's make a prediction.  Here, we set type = "response" to get a probability value

new.x = list(crl.tot = 10, dollar = 0, bang = 3, money = 1, n000 = 0, make = 1)

predict(logistic_model, newdata = new.x, type = "response" )

### Let's get the "confusion matrix" for our training error rates.
### I will use a cut off probability of 50% to classify the
### email has spam or not spam.

predicted_prob = predict(logistic_model, type = "response")

hist(predicted_prob, col="grey")

predicted_yesno_50 = ifelse(predicted_prob >= 0.5, "y", "n")

table(spam7$yesno, predicted_yesno_50)

### Suppose now I use a different threshold probability for my logistic regression: 0, 25, 50, 100

predicted_yesno_00  = ifelse(predicted_prob >= 0.00, "y", "n")
predicted_yesno_25  = ifelse(predicted_prob >= 0.25, "y", "n")
predicted_yesno_75  = ifelse(predicted_prob >= 0.75, "y", "n")
predicted_yesno_100 = ifelse(predicted_prob >= 1.00, "y", "n")

table(spam7$yesno, predicted_yesno_00)
table(spam7$yesno, predicted_yesno_25)
table(spam7$yesno, predicted_yesno_50)
table(spam7$yesno, predicted_yesno_75)
table(spam7$yesno, predicted_yesno_100)

### In this section, I am going to get the AUC for the logistic model.

### There are a few R libraries that create ROC curves. I will use pROC library.  

install.packages("pROC")
library(pROC)

### The option plot = T, gives you the ROC curve but labels the graphs differently
### I will re-plot the ROC for the main model separately

logistic_roc = roc(response = spam7$yesno, predictor = predicted_prob  , plot = T)
logistic_roc$auc

### What if I randomly guessed?  Said all zero?  Said all ones?

set.seed(1000)

n = nrow(spam7)

random_vals = runif(n)
hist(random_vals, col = "grey")

roc(response = spam7$yesno, predictor = random_vals , plot=T)
roc(response = spam7$yesno, predictor = rep(0,n)    , plot=T)
roc(response = spam7$yesno, predictor = rep(1,n)    , plot=T)

### Make a nice plot:

x = seq(from = 0, to = 1, by = 0.01)
y = seq(from = 0, to = 1, by = 0.01)
par(mar=c(6, 6, 4, 4) + 0.1)
plot(y ~ x, xlab="False Positive Rate", ylab="True Positive Rate", type="n", cex.axis = 1.5, cex.lab=1.5)
abline(a=0,b=1,lty=2)
abline(h=0)
lines(1-logistic_roc$specificities, logistic_roc$sensitivities, col="red",     lwd=2)
title(paste("Training AUC = ", round(logistic_roc$auc,3) ))


### Lets play around with some three way interaction logit models 

bad_model = glm(yesno ~ (.)^3 ,  data = spam7, family = binomial)

bad_model$converged

bad_model = glm(yesno ~ (.)^3 ,  data = spam7, family = binomial, control = glm.control(maxit = 5000))

bad_predicted_prob = predict(bad_model, type = "response")

roc(response = spam7$yesno, predictor = bad_predicted_prob  , plot = T)