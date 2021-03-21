#### This is an example of how to use PCA & perform PCA regression analysis ####

#install.packages("faraway")
library(faraway)

### Let's first look at the data we will use:

?meatspec

dim(meatspec)
head(meatspec)

### Our response variable of interest is fat.

### To get an idea about the structure of the data, let's look at a heat map of the correlation matrix.
### But let's look a data that has no correlation.

### Base R can produce "heatmaps" of the data correlation but the following package is nicer.

# install.packages("corrplot")
library(corrplot)

?corrplot
M <- cor(mtcars)
corrplot(M, method = "number")

### Explain what a correlation matrix and plot are and run a few examples.

### To calibrate our eyes, let's look at the correlation plot of 
###  independently distributed data.

n_row = nrow(meatspec)
n_row
n_col = ncol(meatspec)
n_col

set.seed(200)
simulated_data = matrix(rnorm(n_row*n_col), n_row, n_col)

head(simulated_data)

cor_sim = cor(simulated_data)

dim(cor_sim)
cor_sim[1:10,1:10]

### Describe the correlation matrix.

hist(cor_sim[ upper.tri(cor_sim, diag = FALSE) ], col = "grey")

### Just by chance, we some larger correlations (near +0.2 and -0.2).

### Why do you think the output of the following is?

identical(t(cor_sim), cor_sim)

### Here come the plots:

cor_meatspec = cor(meatspec)
cor_meatspec[1:10,1:10]

par(mfrow=c(1,2))

corrplot(cor_sim,      tl.cex = 0.25, type = "upper")
corrplot(cor_meatspec, tl.cex = 0.25, type = "upper")

par(mfrow=c(1,1))

### Let's look some smaller parts:

corrplot(cor_meatspec[1:10,1:10])

cor_meatspec[,n_col]

hist(cor_meatspec[,n_col], col = "grey")

### Let's look at a 6  plots chosen at random.

set.seed(777)
k = sample(1:(n_col-1), size = 6, replace = F)
k

par(mfrow=c(2,3))
for (i in 1:6) 
  plot(fat ~ meatspec[,i], data = meatspec, pch = 19, xlab = k[i], ylab = "fat")
par(mfrow=c(1,1))

par(mfrow=c(2,3))
for (i in 1:6) 
  hist(meatspec[,i], col = "grey", main ="", xlab = k[i])
par(mfrow=c(1,1))

hist(meatspec$fat, col = "grey")

### This data is a great candidate for PCA: highly correlated data.
### To start, let's just see if dimension reduction is possible.  We'll ignore
###  the fat variable: I take out the y variable which is fat in the 101 column.

meat_pca = prcomp(meatspec[,-101], scale = TRUE)
summary(meat_pca)

### The first few components contain nearly all of the variation in the data!

### Let's look at the "screeplot":

screeplot(meat_pca)

component = 1:(n_col-1)
variance  = (meat_pca$sdev)^2
plot(variance ~ component, xlab = "Principal Component", ylab = "Variance", type = "b", pch = 19, col = "red")

plot(variance[1:10] ~ component[1:10], xlab = "Principal Component", ylab = "Variance", type = "b", pch = 19, col = "red")

### Seems like just 2 PC comps are needed
### Now, let's do PCR = Principal Component Regression.
### We'll use another package that does PCR much easier.

# install.packages("pls")
library(pls)

### The default for scaling is FALSE so let's change it to TRUE.
### The default number of folds is 10 and we'll leave that alone.

set.seed(123)
pcr_model = pcr(fat ~ ., data = meatspec, scale = TRUE, validation = "CV", ncomp = n_col - 1)
pcr_cv = RMSEP(pcr_model, estimate = "CV")

pcr_cv

### Let's plot it.  The [-1] leaves out the intercept only model.

plot(pcr_cv$val[-1], pch = 19, type = "b", ylab = "Test RMSE", xlab = "Number of Components")

best_comp = which.min(pcr_cv$val[-1])
best_comp

abline(v = best_comp, col = "red")

pcr_cv$val[ best_comp ]

### Is this a good test rmse?
### Just a rule of thumb: compute coefficient of variation.
### This measures the % error relative to the mean of fat:

mean(meatspec[,n_col])
sd(meatspec[,n_col])

pcr_cv$val[ best_comp ] / mean(meatspec[,n_col]) 

### Suppose you get a new set (5) of observations.  How would you predict the fat content on these?
### We can use the predict function but the variable names need to match.
### For the sake of easiness, let's take the first 5 rows to be "new" observation:
###
### newdata = meatspec[1:5,-n_col] 
###
###
### What would be the predicted fat content?

predict(pcr_model, newdata = meatspec[1:5,-n_col],ncomp = best_comp)

meatspec[1:5, n_col]