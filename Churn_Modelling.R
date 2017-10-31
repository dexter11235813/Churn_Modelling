library(pROC)
library(caret)
library(dplyr)
library(keras)
library(DMwR)
library(parallel)
library(doParallel)


data = read.csv("~/Desktop/Churn_Modelling.csv")
head(data)
data$RowNumber = NULL
data$CustomerId = NULL
data$Surname = NULL

# converting categorical data to 0/1 format
levels(data$Geography) = c(0,1,2)
levels(data$Gender) = c(0,1)

data$NumOfProducts = as.factor(data$NumOfProducts)
data$HasCrCard = as.factor(data$HasCrCard)
data$IsActiveMember = as.factor(data$IsActiveMember)


# creating dummy variables for all categorical data
one_hot = dummyVars(~NumOfProducts + HasCrCard + IsActiveMember + Gender + Geography, data = data)
one_hot_var = predict(one_hot,data)
data[,c("NumOfProducts","HasCrCard","IsActiveMember", "Gender", "Geography")] = NULL
data = cbind(data,one_hot_var)

data = select(data,c(setdiff(1:ncol(data), which(names(data) == "Exited")),which(names(data) == "Exited")))

# Scaling and centering numerical data 
pre = preProcess(data[,c("CreditScore","Balance","EstimatedSalary")], method = c("center","scale"))
data= predict(pre,data)

data$Exited = as.factor(data$Exited)

# balancing data by oversampling from the under-represented target label and undersampling from the
# over-represented target label
data = SMOTE(Exited ~ ., data = data, perc.over = 100, perc.under = 300)
prop.table(table(data$Exited))

levels(data$Exited) = c(0,1)
data$Exited = as.integer(data$Exited)
data = data[sample(1:nrow(data)),]

# creating train/test dataset
idx <- sample(seq(1,2), size = nrow(data), replace = TRUE, prob = c(.7,.3))
train = data[idx == 1,]
test = data[idx == 2, ]

# ANN model

train$Exited = as.factor(train$Exited)
test$Exited = as.factor(test$Exited)
train_x = as.matrix(train[,-19])
train_y = to_categorical(train[,19])
train_y = train_y[,-1]
test_x = as.matrix(test[,-19])
test_y = to_categorical(test[,19])
test_y = test_y[,-1]


model = keras_model_sequential()

model %>% layer_dense(units = 128, input_shape = 18, activation = "relu") %>%
  layer_dense(units = 32,activation = "relu") %>% 
  layer_dropout(.3) %>% 
  layer_dense(2, activation = "softmax")

model %>% compile(loss = "categorical_crossentropy", metric = "accuracy", optimizer = "adam")

model %>% fit(train_x,train_y, batch = 128,validation_split = .2, epoch = 500)

model %>% evaluate(test_x,test_y, batch = 128)

convert = function(x){
  if(x[1] >= x[2]) return(1)
  else return(2)
}

p = predict(model,test_x)
pred = sapply(1:nrow(test_x), function(x) convert(p[x,]))
  

auc(roc(as.numeric(pred),as.numeric(test[,19])))
# 


#  Random forest and Gradient Boosting Machine
  # Creating a dataframe to hold the predicted output of the models
  level.1 = data.frame(pred.GBM = integer(nrow(test)), pred.XGB = integer(nrow(test)))
  cluster <- makeCluster(detectCores() - 1)
  registerDoParallel(cluster)
  t = Sys.time()
  fitControl <- trainControl(method = "repeatedcv",
                             number = 10,
                             repeats = 3,search = "grid",allowParallel = T)
  tunegrid = expand.grid(.mtry = c(1:18))
  train$Exited = as.factor(train$Exited)
  rf.fit = train(Exited ~., data = train, trControl = fitControl,tuneGrid = tunegrid,method = "rf")
  #j = tuneRF(train[,-19],train[,19],stepFactor = 2, improve=1e-5, ntree=500)
  
  p = predict(rf.fit, (test[,-19]))
  auc(roc(as.integer(p), as.integer(test[,17])))
  
  
  gbm.fit = train(Exited ~ ., data = train, trControl = fitControl,method = "gbm")
  pred.fit = predict(gbm.fit,test[,-19])
  
  stopCluster(cluster)
  registerDoSEQ()
  Sys.time() -t 
  
level.1$pred.GBM = as.integer(pred.fit) - 1
level.1$pred.RF = as.integer(p)


# Extreme Gradient Boosting
library(xgboost)
  
train$Exited = as.integer(train$Exited) - 1
test$Exited = as.integer(test$Exited) - 1
dtrain = xgb.DMatrix(data = as.matrix(train[,-19]), label = (train[,19]))
dtest = xgb.DMatrix(data = as.matrix(test[,-19]), label = (test[,19]))

params <- list(booster = "gbtree", objective = "binary:logistic", eta=0.3, gamma=0, max_depth=6, min_child_weight=1, subsample=1, colsample_bytree=1)

xgbcv <- xgb.cv( params = params, data = dtrain, nrounds = 100, nfold = 5, showsd = T, stratified = T, print_every_n = 10, early_stop_round = 20, maximize = F)


xgb1 <- xgb.train (params = params, data = dtrain, nrounds = 90, watchlist = list(val=dtest,train=dtrain), print_every_n = 10, early_stop_round = 10, maximize = F , eval_metric = "error")

pred.xgb = predict(xgb1,dtest)
pred.xgb = ifelse(pred.xgb > .5,1,0)
auc(roc(pred.xgb, test[,19]-1))

level.1$pred.XGB = pred.xgb


# Adaboost :
# 
# train1 = train
# test1 = test
# levels(train1$Exited) = c("No","Yes")
# levels(test1$Exited) = c("No","Yes")
# test1$Exited = as.factor(test1$Exited)
# train1$Exited = as.factor(train1$Exited)
# fit.ADA = train(Exited ~., train1, trControl = fitControl, method = "adaboost")
# pred.ADA = predict(fit.ADA, test1[,-19])
# level.1$pred.ADA = as.factor(pred.ADA)

# SVM
library(e1071)
train$Exited = as.factor(train$Exited)
fit.svm = svm(Exited ~., data = train,gamma = 1, cost = 4)
pred.svm = predict(fit.svm,test[,-19])
#j = tune(svm,Exited ~.,data = train,ranges = list(gamma = 2^(-2:2),cost = 2^(2:4)),tunecontrol = tune.control(cross = 10))
auc(roc(as.numeric(pred.svm),as.numeric(as.factor(test[,19]))))

level.1$pred.SVM = pred.svm
level.1$pred.RF = as.factor(level.1$pred.RF)
level.1$pred.GBM = as.factor(level.1$pred.GBM)
level.1$pred.XGB = as.factor(level.1$pred.XGB)
level.1$predNN = as.factor(pred - 1)
level.1$label =  as.factor(test$Exited)


# Now that I have predictions from all the models, I will split the level.1 dataset into train and test dataset to
# train a model to predict the label column from the predictions of all other models.
ind = sample(1:nrow(level.1),.7*nrow(level.1))
level.1.train = level.1[ind,]
level.1.valid = level.1[-ind,]

level.1.fit = train(label ~.,data = level.1.train,trControl = fitControl, method = "glm")
pred.val = predict(level.1.fit,level.1.valid[,-ncol(level.1.valid)])

auc = roc(as.numeric(pred.val), as.numeric(level.1.valid[,ncol(level.1.valid)]))
auc(auc)


