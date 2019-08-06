---
title: "Practical Machine Learning"
author: "H.G"
date: "8/4/2019"
output: html_document
keep_md: TRUE
---

## Overview
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, who were asked to perform barbell lifts correctly and incorrectly in 5 different ways. And predict the manner in which they did the exercise.

## Setting Up
Load required packages:

```r
rm(list=ls())
suppressWarnings(library(caret, warn.conflicts = FALSE))
suppressWarnings(library(randomForest, warn.conflicts = FALSE))
suppressWarnings(library(e1071, warn.conflicts = FALSE))
suppressWarnings(library(dplyr, warn.conflicts = FALSE))
```

## Loading Data

```r
train <- read.csv("./pml-training.csv",header = TRUE, na.strings = c ("","NA"))

test <- read.csv("./pml-testing.csv",header = TRUE, na.strings = c ("","NA"))
dim(test)
```

```
## [1]  20 160
```

```r
dim(train)
```

```
## [1] 19622   160
```



```r
train <- mutate(train, NAcounter=rowSums(is.na(train))) 
train <- train[which(train$NAcounter != 0),]
train <- train[,-c(161)]

# Removing near zero values
nzvInTrain <- nearZeroVar(train)
train <- train[,-nzvInTrain]

# Remove participant identifiers in each row and timestamps
train <- train[, -c(1:5)]

## TESTING DATASET
# Cleaning NA variables
test <- mutate(test, NAcounter=rowSums(is.na(test))) 
test <- test[which(test$NAcounter != 0),]
test <- test[,-c(161)]

# Removing near zero values
nzvInTest <- nearZeroVar(test)
test <- test[,-nzvInTest]

# Remove participant identifiers in each row and timestamps
test <- test[, -c(1:5)]
```



We can see the testing dataset contains 20 observations and 160 variables, and the training dataset contains 19622 observations and 160 variables. The last column "classe" will be the goal to be predicted.

## Data Preprocessing
The first 5 variables are just used for identification, thus remove them. In addition, columns that contains too many zero values are being removed.

```r
train <- train[,-(1:5)]
test <- test[,-(1:5)]
train <- train[, colSums(is.na(train)) == 0] 
test <- test[, colSums(is.na(test)) == 0] 
```

## Data Partitions
Split the training dataset into 70:30 (training:validation)

```r
set.seed(13434)
inTrain<-createDataPartition(y=train$classe, p=0.7, list=FALSE)
training<-train[inTrain, ]
testing<-train[-inTrain, ]
```
## Random Forest
Fitting a random forest predictive model and using 3-fold cross-validation.

```r
fitRF <- train(classe ~ ., method = "rf", data = training, number=3, ntree=200)
```

```r
pdRF <- predict(fitRF, testing)
cmRF <- confusionMatrix(testing$classe,pdRF)
cmRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1632    7    0    1    1
##          B   14 1094    6    1    0
##          C    0   12  991    2    0
##          D    0    0   30  913    1
##          E    0    0    2    1 1055
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9865          
##                  95% CI : (0.9831, 0.9893)
##     No Information Rate : 0.2856          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9829          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9915   0.9829   0.9631   0.9946   0.9981
## Specificity            0.9978   0.9955   0.9970   0.9936   0.9994
## Pos Pred Value         0.9945   0.9812   0.9861   0.9672   0.9972
## Neg Pred Value         0.9966   0.9959   0.9920   0.9990   0.9996
## Prevalence             0.2856   0.1931   0.1786   0.1593   0.1834
## Detection Rate         0.2832   0.1898   0.1720   0.1584   0.1831
## Detection Prevalence   0.2847   0.1935   0.1744   0.1638   0.1836
## Balanced Accuracy      0.9947   0.9892   0.9801   0.9941   0.9987
```

```r
accuracy <- postResample(pdRF, testing$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9864654 0.9828763
```
So the esimated accuracy is 98.6465383%, and the estimated out-of-sample error is 1.3534617%

## Apply to the 20 Test Cases
Applying the model we fit above to the 20 test cases and see the results.

```r
result <- predict(fitRF, test)
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
















