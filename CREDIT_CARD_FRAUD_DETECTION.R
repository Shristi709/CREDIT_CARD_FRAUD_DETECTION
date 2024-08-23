# Build a machine learning model to identify fraudulent credit card
#transactions.

rm(list = ls())
# Loading necessary libraries
library(caret)
library(dplyr)
library(ggplot2)
library(randomForest)
library(e1071)  # For logistic regression
library(pROC)   # For ROC and AUC

data=read.csv("C:/Users/shris/Downloads/creditcard.csv")

# Exploring the dataset
View(data)
str(data)
summary(data)





# PREPROCESSING AND NORMALIZING THE DATA

# Checking for missing values
colSums(is.na(data))

# Normalizing
data$Amount=scale(data$Amount)
data$Time=scale(data$Time)

# Dropping irrelevant columns if necessary 
data=data %>% select(-Time)

# Dimension of the data
dim(data)





# VISUALISATION

# Visualising distribution of the Target Variable
table(data$Class)  # Count of fraudulent vs non-fraudulent transactions

ggplot(data, aes(x = factor(Class))) +
  geom_bar(fill = c("skyblue", "red")) +
  labs(title = "Distribution of Fraudulent vs Non-Fraudulent Transactions",
       x = "Class (0 = Non-Fraud, 1 = Fraud)",
       y = "Count")

# Correlation Matrix
corr_matrix=cor(data[,-c(1, ncol(data))])  # Excluding 'Time' and 'Class'
library(corrplot)
corrplot(corr_matrix, method = "color", tl.cex = 0.7)





# MODELING

# Splitting the data into training and testing Sets (80-20 split)
set.seed(1234)
n=nrow(data)  # Number of rows in the dataset
train_size=floor(0.8 * n)  # Determine the number of rows for the training set (80%)
i=sample(1:nrow(data),size = train_size)
train=data[i,]
test=data[-i,]

dim(train);dim(test)

# Checking the distribution in the training and test sets
table(train$Class)
table(test$Class)


# Logistic Regression
lr_model=glm(Class ~ ., family = binomial, data = train)
summary(lr_model)

# Predicting on the test data
lr_predictions=predict(lr_model, newdata=test, type = 'response')
lr_pred_class=ifelse(lr_predictions > 0.5, 1, 0)

# Evaluating the logistic regression model
confusionMatrix(factor(lr_pred_class), factor(test$Class))





# EVALUATING MODEL PERFORMANCE

# Logistic Regression: Evaluating with Precision, Recall, F1-score, and AUC
lr_cm=confusionMatrix(factor(lr_pred_class), factor(test$Class))
precision_lr=lr_cm$byClass["Pos Pred Value"]
recall_lr=lr_cm$byClass["Sensitivity"]
f1_lr=2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)

roc_lr=roc(test$Class, lr_predictions)
auc_lr=auc(roc_lr)

metrics_name=c("Precision","Recall","F1 Score","AUC")
metrics_result=c(precision_lr,recall_lr,f1_lr,auc_lr)
d=data.frame(metrics_name,metrics_result);d



























