
#libraries
library(vroom) #read in data
library(dplyr); library(ggplot2); library(tidyr); library(purrr); library(reshape2) #manipulate and visualise data
library(Hmisc); library(mice); library(VIM); #impute missing values
library(caret); library(rattle); library(e1071); library(xgboost) #machine learning

#################
#Read in data
#################

unzip("titanic.zip")
gender_sub <- vroom("gender_submission.csv")
test <- vroom("test.csv")
train <- vroom("train.csv")

##########################
#exploritory data analysis
##########################

#What does the data look like?
head(train)
summary(train)


#how many zeros by varible
plot(colMeans(is.na(train)))
#noted that Age has around 20% NA values, while Cabin has around 80% NA values.

#expoloritory plots
#------------------

#how many of the depedent variable
ggplot(data = train, aes(Survived)) +
    geom_bar()

#histogram of all variables
train %>%
    keep(is.numeric) %>%
    gather() %>%
    ggplot(aes(value)) +
    facet_wrap(~ key, scales = "free") +
    geom_histogram()

#plot variables against dependent variables to see if any relationships initially jump out
train %>%
    gather(-Survived, key = "var", value = "value") %>%
    ggplot(aes(x = as.factor(Survived), y = value, colour = as.factor(Survived))) +
    geom_jitter(alpha = 0.5) +
    facet_wrap(~var, scales = "free") +
    theme(axis.text.y = element_blank(), legend.position = "none")

#In particular, "Sex" and "Pclass" seems to have a strong relationship with survival. "Parch" and "SibSp" also seem to display some relationship, although it is less clear 
#Plot all variables against Sex, using colour to show survival or not
train %>%
    gather(-Survived, -Sex, key = "var", value = "value") %>%
    ggplot(aes(x = value, y = as.factor(Sex), colour = as.factor(Survived))) +
    geom_jitter(alpha = 0.5) +
    facet_wrap( ~var, scales = "free") +
    theme(axis.text.x = element_blank(), 
          axis.text.y = element_text(angle = 90),
          legend.position = "none")

#The relationship between "Pclass" and survival is even more clear when controlling for Gender. "Parch" and "SibSp" remain potentially predictive.  
train %>%
    gather(-Survived, -Sex, key = "var", value = "value") %>%
    group_by(var) %>%
    summarise()
    ggplot(aes(x = value, y = as.factor(Sex), colour = as.factor(Survived))) +
    geom_jitter(alpha = 0.5) +
    facet_wrap( ~var, scales = "free") +
    theme(axis.text.x = element_blank(), 
          axis.text.y = element_text(angle = 90),
          legend.position = "none")

#plotting percentage of survival against variables, controlling for gender
train %>%
    mutate(Age = findInterval(Age, seq(0,100,5)), Fare = findInterval(Fare, seq(0,540,20))) %>%
    select(-c(Name, PassengerId)) %>%
    gather(-Survived, -Sex, key = "var", value = "value") %>%
    group_by(var, value, Sex) %>%
    summarise(survival = sum(Survived)/n()) %>%
    ggplot(aes(x = value, y = survival, colour = Sex)) +
    geom_point() +
    geom_smooth(se = FALSE) +
    facet_wrap(~var, scales = "free") +
    theme(axis.text.x = element_blank(),
          legend.position = "bottom")

#specify factor variables
factorVars <- c("Survived", "Pclass", "Sex", "Embarked")
train[factorVars] <- lapply(train[factorVars], as.factor)

#exploring the "Ticket" variable a little more:
train %>%
    group_by(Ticket) %>%
    summarise(frequency = n()) %>%
    arrange(desc(frequency))

#ticket is clearly not a unique identifier. I examine the two most frequent tickets to see if there is seems to be a relationship between tickets of the same value
train %>%
    filter(Ticket %in% c("1601","347082")) %>%
    arrange(Ticket)

#due to the nature of the variable (categorical) and the number of unique values, I will remove the  variable at this time

#Dealing with NA values
#---------------------
#see where there are missing values
aggr(x = train)

#mean/median, knn approach
trainNAChoice <- train %>%
    mutate(AgeNAMean = impute(Age, mean),
           AgeNAMedian = impute(Age, median),
           AgeNAMice = complete(mice(data = train, method = "pmm"), 1)$Age,
           AgeNAKnn = kNN(data = train, variable = c("Age"))$Age)

#seeing distrubtion of age based on imputation method
ggplot(data = trainNAChoice %>%
           select(Age, AgeNAMean, AgeNAMedian, AgeNAMice, AgeNAKnn) %>%
           melt(variable.name = "NAImputeMethod", variable.value = "Age" ),
       aes(x = value)) +
    geom_histogram() +
    facet_wrap(~NAImputeMethod)

#based on the distributions using the above methods, I will use the knn approach for missing values


#Selecting variables
#-------------------

#As per the previous section, over 70% of the values are missing for the "Cabin" variable, so I will drop this variable
train <- select(train, -c(Cabin, Ticket))

#Do any variables have near zero variance? As per below, no.
nearZeroVar(train)

#I will drop the identifier variables, "Name" and "PassengerId"
train <- select(train, -c(PassengerId, Name))

#fill in the missing values based on knn, not using the dependent variable
    trainFinal <- kNN(data = train,
                  dist_var = colnames(train)[!colnames(train) %in% c("Survived")],
                  imp_var = FALSE)

##########
#Modelling
##########

## models? - logistic regression, KNN, Decision trees, random forest, xgboost, neural network?

#Regression Model
#----------------

#first model, logistic regression with only  the four most obvious variables, with 5-fold Cross validation
modLogitBasic <- train(Survived ~ Sex + Parch + SibSp + Pclass, data = trainFinal,
                  method = "glm", family = "binomial",
                  trControl = trainControl(method = "cv", number = 5))
summary(modLogitBasic)
modLogitBasic

#We get 79% accuracy on the training set. With the Sex of the passanger and class particurly predictive
# The number of siblings abroad is alos slightly predictive 

modLogitAll <- train(Survived ~ ., data = trainFinal,
                     method = "glm", family = "binomial",
                     trControl = trainControl(method = "cv", number = 5))
summary(modLogitAll)
modLogitAll

#with all the relevant variables, we get 80% accuracy on the training set

#Knn
#---
modKnn <- train(Survived ~ ., data = trainFinal,
                method = "knn",
                trControl = trainControl(method = "cv", number = 5))
modKnn
#We get 71% accuracy on the training set

#Decision Tree-based models
#--------------

#Basic decision tree
modDTree <- train(Survived ~ ., data = trainFinal,
                  method = "rpart",
                  trControl = trainControl(method = "cv", number = 5))
modDTree
fancyRpartPlot(modDTree$finalModel)
#we get 81% accuracy on the training set

#random forest
modRF <- train(Survived ~ ., data = trainFinal,
               method = "rf",
               trControl = trainControl(method = "cv", number = 5))
modRF
modRF$finalModel
#we get 83.5% accuracy on the training set

#xgboost
#convert data to correct format
labels <- trainFinal$Survived
trainXGS <- model.matrix(~.+0, select(trainFinal, -c(Survived)))
labels <- as.numeric(labels) - 1
dtrain <- xgb.DMatrix(data = trainXGS, label = labels)

#model
params <- list(booster = "gbtree",
               objective = "binary:logistic",
               eta = 0.3,
               gamma = 0,
               max_depth = 6,
               min_child_weigth = 1,
               subsample = 1,
               colsample_bytree = 1)

xgbcv <- xgb.cv(params = params, 
                data = dtrain, 
                nrounds = 100, 
                nfold = 5, 
                showsd = TRUE, 
                stratified = TRUE, 
                print_every_n = 10, 
                early_stopping_rounds = 20, 
                maximize = FALSE)
xgbcv
modXGB1 <- xgb.train(params = params,
                     data = dtrain,
                     nrounds = 200,
                     watchlist = list(train = dtrain),
                     print_every_n = 10, 
                     early_stopping_rounds = 10,
                     maximize = FALSE,
                     eval_metric = "error")
modXGB1
mat <- xgb.importance((feature_names = colnames(trainXGS)), model = modXGB1)
xgb.plot.importance(importance_matrix = mat)

predXgb1Train <- predict(modXGB1, 
                         newdata = dtrain)
acc <- mean(labels == ifelse(predXgb1Train >= 0.5, 1, 0))
#xgboost has 96.7% accuracy on the training data

#Naive Bayes
#-----------

modNB <- train(Survived ~ ., data = trainFinal,
               method = "nb",
               trControl = trainControl(method = "cv", number = 5))
modNB
#Naive Bayes has a 78% Accuracy rate on the training data


