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
##          A 1674    0    0    0    0
##          B    1 1136    2    0    0
##          C    0    3 1023    0    0
##          D    0    0    5  958    1
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2846          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   0.9974   0.9932   1.0000   0.9991
## Specificity            1.0000   0.9994   0.9994   0.9988   1.0000
## Pos Pred Value         1.0000   0.9974   0.9971   0.9938   1.0000
## Neg Pred Value         0.9998   0.9994   0.9986   1.0000   0.9998
## Prevalence             0.2846   0.1935   0.1750   0.1628   0.1840
## Detection Rate         0.2845   0.1930   0.1738   0.1628   0.1839
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9984   0.9963   0.9994   0.9995
```

```r
accuracy <- postResample(pdRF, testing$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9979609 0.9974207
```
So the esimated accuracy is 99.7960918%, and the estimated out-of-sample error is 0.2039082%

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
















