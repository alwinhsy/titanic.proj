###############################
#code to restart your console##
###############################
rm(list =ls())
cat("\014")

#################
#load libraries
#################
library(glmnet)
library(caret)
library(tree)
library(rpart)
library(randomForest)
library(gbm)
library(ROCR)
library(dplyr)
library(nnet)
library(class)
library(sjmisc)

#start
setwd("C:/Users/Alwin/Downloads/Titanic Project")

train = read.csv("train.csv")
test = read.csv("test.csv")

#Data cleaning
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
#data imputation
Mode(train$Embarked)
Mode(train$Parch)
mean(train$Age, na.rm = TRUE)
mean(train$Fare, na.rm = TRUE)

train$Age[is.na(train$Age)] = 29.69912
test$Age[is.na(test$Age)] = 29.69912
test$Fare[is.na(test$Fare)] = mean(train$Fare, na.rm = TRUE)
test$Parch[test$Parch == '9'] = 0
train$Embarked[train$Embarked == ''] = "S"

train2 = subset(train, select=-c(PassengerId, Name, Ticket, Cabin))
test2 = subset(test, select=-c(PassengerId, Name, Ticket, Cabin))

#change all to factor
train2$Pclass = factor(train2$Pclass)
train2$Sex = factor(train2$Sex)
train2$SibSp = factor(train2$SibSp)
train2$Parch = factor(train2$Parch)
train2$Embarked = factor(train2$Embarked)
train2$Survived = factor(train2$Survived)

test2$Pclass = factor(test2$Pclass)
test2$Sex = factor(test2$Sex)
test2$SibSp = factor(test2$SibSp)
test2$Parch = factor(test2$Parch)
test2$Embarked = factor(test2$Embarked)

#results vector to compare
y.factor.test = factor(read.csv("gender_submission.csv")[,2])





############
##Logistic##
############

log = glm(Survived~ ., data = train2, family = binomial(link = "logit"))
log.pred = predict(log, newdata = test2, type = "response")
View(log.pred)

log.pred.factor <- log.pred
log.pred.factor[log.pred.factor > 0.5] <- 1
log.pred.factor[log.pred.factor < 0.5] <- 0
log.pred.factor <-  factor(log.pred.factor) # labeling prediction as factor levels

confusionMatrix(log.pred.factor, y.factor.test, positive = '1')
#0.9472

############
##Tree-based###
###############

#simple tree
set.seed(12345)
big.tree = rpart(Survived~ .,method="class",data=train2)
length(unique(big.tree$where)) #7
par(mfrow=c(1,1)) #back to one graph per window
plotcp(big.tree) 
bestcp=big.tree$cptable[which.min(big.tree$cptable[,"xerror"]),"CP"]
best.tree = prune(big.tree,cp=bestcp) 

simpletree.pred <- predict(best.tree, newdata = test2)
simpletree.pred.factor <- simpletree.pred
simpletree.pred.factor[simpletree.pred.factor > 0.5] <- 1
simpletree.pred.factor[simpletree.pred.factor < 0.5] <- 0
simpletree.pred.factor = simpletree.pred.factor[,2]
simpletree.pred.factor = factor(simpletree.pred.factor)

confusionMatrix(simpletree.pred.factor, y.factor.test, positive = "1")
#0.9713  

#BAGGING
set.seed(12345)
bagfit = randomForest(Survived~.,data=train2,ntree=5000,maxnodes=length(unique(best.tree$where)), mtry= 7)

bagfit.pred.prob = predict(bagfit, newdata=test, type = "class")
bagfit.pred.factor <- bagfit.pred.prob
bagfit.pred.factor[bagfit.pred.factor > 0.5] <- 1
bagfit.pred.factor[bagfit.pred.factor < 0.5] <- 0
bagfit.pred.factor <-  factor(bagfit.pred.factor) 

confusionMatrix(bagfit.pred.factor, y.factor.test, positive = "1")
#0.9909

#randomforests
set.seed(12345)
rffit = randomForest(Survived~.,data=train2,ntree=5000,maxnodes=10, mtry= ceiling(sqrt(7)))

rffit.pred.prob = predict(rffit, newdata=test, type = "class")
rffit.pred.factor <- rffit.pred.prob
rffit.pred.factor[rffit.pred.factor > 0.5] <- 1
rffit.pred.factor[rffit.pred.factor < 0.5] <- 0
rffit.pred.factor <-  factor(rffit.pred.factor)
confusionMatrix(rffit.pred.factor, y.factor.test, positive = "1")
# 0.9305




#################
##simple ANN##
##################

#create dummies
dummies = to_dummy(subset(train, select=c(Sex, Embarked)), var.name = "name", suffix = c("numeric", "label"))
trainann = subset(train, select=-c(PassengerId, Name, Ticket, Cabin, Sex, Embarked))
trainann = data.frame(trainann, dummies)
dummies.test = to_dummy(subset(test, select=c(Sex, Embarked)), var.name = "name", suffix = c("numeric", "label"))
testann = subset(test, select=-c(PassengerId, Name, Ticket, Cabin, Sex, Embarked))
testann = data.frame(testann, dummies.test)

#Perform the min-max normalization":
maxs = apply(trainann, 2, max) #find maxima by columns (i.e., for each variable)
mins = apply(trainann, 2, min) #find minima by columns (i.e., for each variable)

#Perform the min-max normalization":
maxs2 = apply(testann, 2, max) #find maxima by columns (i.e., for each variable)
mins2 = apply(testann, 2, min) #find minima by columns (i.e., for each variable)

#Do the normalization. Note scale subtracts the "center" parameter and divides by "scale" parameter:
trainscaled = as.data.frame(scale(trainann, center = mins, scale = maxs - mins))
testscaled = as.data.frame(scale(testann, center = mins2, scale = maxs2 - mins2))

trainann$Survived = as.factor(trainann$Survived)

start.time <- Sys.time()
set.seed(12345)
ann.fit <- train(Survived ~ ., data = trainscaled,
                 method = "nnet", maxit = 1000, trace = F, linout = F,  tuneGrid = expand.grid(.size=c(1,2,3,4,5,6,7,8,9,10),.decay=c(0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10)))
end.time <- Sys.time()
end.time-start.time #Time difference of 9.341857 mins

nn=nnet(Survived~., data=trainscaled, size=2,  maxit=1000, decay=0.03, linout = F, trace= F)
nn.pred <- predict(nn, testscaled, type ='raw')
nn.pred.factor <- nn.pred
nn.pred.factor[nn.pred.factor > 0.5] <- 1
nn.pred.factor[nn.pred.factor < 0.5] <- 0
nn.pred.factor <-  factor(nn.pred.factor) 

confusionMatrix(as.factor(nn.pred.factor), y.factor.test, positive = "1")
#0.8657

######
#combination
###########
average.pred = (as.vector(log.pred) + as.vector(simpletree.pred[,2]) + as.vector(nn.pred))/3
average.pred.factor <- average.pred
average.pred.factor[average.pred.factor > 0.5] <- 1
average.pred.factor[average.pred.factor < 0.5] <- 0
average.pred.factor <-  factor(average.pred.factor) 
confusionMatrix(as.factor(average.pred.factor), y.factor.test, positive = "1")


majority.pred = as.factor(ifelse(log.pred.factor == 1 & simpletree.pred.factor == 1, 1,
                                 ifelse(log.pred.factor == 1 & nn.pred.factor == 1, 1,
                                        ifelse(simpletree.pred.factor == 1 & nn.pred.factor == 1, 1, 0))))

confusionMatrix(majority.pred, y.factor.test, positive = "1")