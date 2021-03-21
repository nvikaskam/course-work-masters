### The data comes from insurance industry
### The full description of the data can be found at:
###  http://instruction.bus.wisc.edu/jfrees/jfreesbooks/Regression%20Modeling/BookWebDec2010/DataDescriptions.pdf
###
### Our variables we will work with are:
###  (1) FACE      = Amount that the company will pay in the event of the death of the named insured (This is our Y.)
###  (2) INCOME    = Annual income of the family
###  (3) EDUCATION = Number of years of education of the survey respondent
###  (4) NUMHH     = Number of household members
###  (5) MARSTAT   = Marital status of the survey respondent: 1 if married, 2 if living with partner, and 0 otherwise

original.term.life.data = read.csv("TermLife.csv", header = T)
head(original.term.life.data)

### Take the subset of the data.  I am only interested in data with FACE > 0

term.life.data = original.term.life.data[  original.term.life.data$FACE > 0,   c('FACE', 'INCOME', 'EDUCATION', 'NUMHH', 'MARSTAT')]
head(term.life.data)

term.life.data$FACE      = log(term.life.data$FACE)
term.life.data$INCOME    = log(term.life.data$INCOME)
term.life.data$MARSTAT   = factor(term.life.data$MARSTAT, levels = c(0,1,2), labels = c("Other","married","partnered"))
colnames(term.life.data) = c("logPayment","logIncome","Education","NoHousholds","MStat")

### Done pre-processing.  Let's look at the data.

head(term.life.data)
str(term.life.data)
summary(term.life.data)

### Now do the analysis

install.packages("rpart")
library(rpart)

termlife.tree.model = rpart(logPayment ~ ., data = term.life.data)
termlife.tree.model

### Plot the tree:

plot(termlife.tree.model)
text(termlife.tree.model)

### But could make it nicer.  

plot(termlife.tree.model, uniform=T,branch=0.2,margin=0.1)
text(termlife.tree.model)

install.packages("rpart.plot")
library(rpart.plot)
windows()
rpart.plot(termlife.tree.model, digits = 3)

### Let us verify the prediction we made:

new.observation = list(logIncome = log(45000), Education = 12,  NoHousholds = 4,   MStat = "married")

predict(termlife.tree.model, newdata = new.observation)

exp(predict(termlife.tree.model, newdata = new.observation))

### Lets try the rpart algo on the spam data set from DAAG package

install.packages("DAAG")
library(DAAG)

### Build the tree

spam.tree.model = rpart(yesno ~ ., data = spam7)
spam.tree.model

### Plot the tree.

plot(spam.tree.model)
text(spam.tree.model)

### Prediction:

new.observation.spam = list( crl.tot = 10, dollar = 0, bang = 3, money = 1, n000 = 0, make = 1)
predict(spam.tree.model, newdata = new.observation.spam )

spam.tree.model