### Stochastic Gradient Descent Code

install.packages("sgd")
library(sgd)

### Let's create a multiple regression model:

set.seed(2017)

n = 100
n = 10^4

x1 = runif(n)
x2 = runif(n)

b1    = 3
b2    = -7
sigma = 4

y  = b1*x1 + b2*x2 + rnorm(n, sd = sigma)

### sgd is written so it is consistent with the way R works.
### Changing the learning rate for this function is complicated so 
###  I will use default.
### The -1 inside the sgd AND lm says that don't use an intercept.

m = sgd(y ~ x1 + x2 - 1, model = "lm")
m
m$converged

### Compare with regular lm

g = lm(y ~ x1 + x2 - 1)
g

### Go back and do it with 10^4

### Let's look at the loss

b1_hat = m$estimates[1,]
b2_hat = m$estimates[2,]

plot(b2_hat ~ b1_hat, pch = 19, type = "b", lwd = 2, xlim = c(-1,4), ylim = c(-8,1))
points(3,-7, col = "red", pch = 19, cex = 1.5)

### The final point doesn't quite get to the red point.

len = length(b1_hat)
loss = numeric(len)

for (i in 1:len)
{
  loss[i] = mean((b1_hat[i] * x1 + b2_hat[i] * x2 - y)^2)
}

windows()
plot(loss, type = "l", xlab = "Iteration", ylab = "MSE")

### Notice that the loss function fluctuated and in fact went up at first
###  before going down.