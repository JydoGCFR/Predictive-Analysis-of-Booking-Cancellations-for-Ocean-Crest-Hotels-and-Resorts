# Required libraries
library(caret)
library(tidyverse)
library(skimr)
library(xgboost)
library(doParallel)
library(SHAPforxgboost)
library(pROC)

# Step 1: Load Data and Drop Specified Columns
hotel_data <- read.csv("OceanCrestModified.csv", header = TRUE)

hotel_data <- hotel_data %>%
  select(-is_repeated_guest, -is_agent, -is_company, -distribution_channel, -babies, -is_meal, -continent)

# Step 2: Data Pre-processing
hotel_data <- hotel_data %>%
  mutate_at(c("is_canceled", "arrival_date_year", "hotel", "market_segment", "deposit_type", "customer_type", "is_reserved_room", "arrival_season"), as.factor)

# Rename and Relevel 'is_canceled'
hotel_data$is_canceled <- fct_recode(hotel_data$is_canceled, cancelled = "1", notcancelled = "0")
hotel_data$is_canceled <- relevel(hotel_data$is_canceled, ref = "cancelled")

# Prepare predictors and response
hotel_predictors_dummy <- model.matrix(is_canceled ~ ., data = hotel_data)
hotel_predictors_dummy <- data.frame(hotel_predictors_dummy[, -1])
hotel_data <- cbind(is_canceled = hotel_data$is_canceled, hotel_predictors_dummy)

# Step 3: Partition the Data (80% training, 20% testing)
set.seed(99)
index <- createDataPartition(hotel_data$is_canceled, p = 0.8, list = FALSE)
hotel_train <- hotel_data[index,]
hotel_test <- hotel_data[-index,]

# Step 4: Start Parallel Processing
num_cores <- detectCores(logical = FALSE)
cl <- makePSOCKcluster(num_cores - 2)
registerDoParallel(cl)

# Step 5: Define the Tuning Grid
tune_grid <- expand.grid(
  nrounds = c(100, 200, 300),        # More boosting iterations
  eta = c(0.01, 0.05, 0.1),          # Lower learning rate
  max_depth = c(3, 6, 9),            # Deeper trees
  gamma = c(0, 1, 5),                # Minimum loss reduction
  colsample_bytree = c(0.6, 0.8, 1), # Fraction of features to sample
  min_child_weight = c(1, 5, 10),    # Minimum sum of instance weight (hessian)
  subsample = c(0.6, 0.8, 1)         # Fraction of samples to use for training
)

# Step 6: TrainControl Setup for Cross-Validation
train_control <- trainControl(
  method = "cv",                    # Cross-validation
  number = 5,                       # 5-fold cross-validation
  classProbs = TRUE,                # Enable AUC calculation
  summaryFunction = twoClassSummary, # Use AUC as metric
  verboseIter = TRUE                # Show progress
)

# Step 7: Train the XGBoost Model with Cross-Validation
set.seed(8)
model_xgb_tuned <- train(
  is_canceled ~ .,                  # Predictors and response
  data = hotel_train,               # Training data
  method = "xgbTree",               # XGBoost method
  tuneGrid = tune_grid,             # Grid search values
  trControl = train_control,        # Cross-validation settings
  metric = "ROC"                    # Optimize for AUC
)

# Stop parallel processing
stopCluster(cl)
registerDoSEQ()

# Step 8: Check the best-tuned parameters
print(model_xgb_tuned$bestTune)

# Step 9: Plot Important Variables and SHAP Analysis
plot(varImp(model_xgb_tuned), top = 10)

# SHAP analysis
Xdata <- as.matrix(select(hotel_train, -is_canceled))
shap <- shap.prep(model_xgb_tuned$finalModel, X_train = Xdata)
shap.plot.summary.wrap1(model_xgb_tuned$finalModel, X = Xdata, top_n = 10)

# Partial Dependence Plot Example
p <- shap.plot.dependence(
  shap, 
  x = "lead_time", 
  color_feature = "adults", 
  smooth = FALSE, 
  jitter_width = 0.01, 
  alpha = 0.4
) + ggtitle("Lead Time vs Adults")
print(p)

# Step 10: Model Evaluation on the Test Set
# Predict probabilities and classes
predictions_prob <- predict(model_xgb_tuned, hotel_test, type = "prob")
predictions <- predict(model_xgb_tuned, hotel_test)

# Confusion matrix
conf_matrix <- confusionMatrix(predictions, hotel_test$is_canceled)
print(conf_matrix)

# ROC curve and AUC on the test set
roc_test <- roc(hotel_test$is_canceled, predictions_prob[, "cancelled"])
plot(roc_test, col = "blue", main = "ROC Curve for Tuned XGBoost Model")
auc_test <- auc(roc_test)
print(paste("AUC on test data: ", auc_test))

# Step 11: Retrieve Cross-Validation Performance (from training phase)
cv_auc <- max(model_xgb_tuned$results$ROC)
print(paste("Cross-validation AUC: ", cv_auc))

# Step 12: Compare Cross-Validation AUC and Test Set AUC
if (cv_auc - auc_test > 0.05) {
  print("Warning: Possible overfitting detected. Significant difference between cross-validation and test set AUC.")
} else {
  print("No significant overfitting detected. AUC values are comparable.")
}








# Other SHAP dependency plots
# SHAP dependence plot for Deposit Type
p1 <- shap.plot.dependence(
  shap, 
  x = "deposit_typeNon.Refund",  # Feature representing deposit type (Non-Refundable)
  color_feature = "lead_time",   # Color by lead time to see interactions
  smooth = FALSE, 
  jitter_width = 0.01, 
  alpha = 0.4
) + ggtitle("Non-Refundable Deposit vs Lead Time")
print(p1)


# SHAP dependence plot for ADR
p2 <- shap.plot.dependence(
  shap, 
  x = "adr",                     # ADR feature (Average Daily Rate)
  color_feature = "market_segment",  # Color by market segment to observe pricing impact by segment
  smooth = FALSE, 
  jitter_width = 0.01, 
  alpha = 0.4
) + ggtitle("ADR vs Market Segment")
print(p2)


# SHAP dependence plot for Special Requests
p3 <- shap.plot.dependence(
  shap, 
  x = "total_of_special_requests",  # Total number of special requests
  color_feature = "lead_time",      # Color by lead time to observe how it interacts with special requests
  smooth = FALSE, 
  jitter_width = 0.01, 
  alpha = 0.4
) + ggtitle("Total Special Requests vs Lead Time")
print(p3)






# Required libraries
library(ggplot2)
library(caret)

# Step 1: Predict probabilities for the training set
predictions_prob_train <- predict(model_xgb_tuned, hotel_train, type = "prob")

# Step 2: Create a new data frame with actual class labels and predicted probabilities
train_data_probs <- data.frame(
  is_canceled = hotel_train$is_canceled,
  predicted_prob = predictions_prob_train[, "cancelled"]  # Use "cancelled" probability
)

# Step 3: Plot the predicted probability distribution
ggplot(train_data_probs, aes(x = predicted_prob, fill = is_canceled, color = is_canceled)) +
  geom_density(alpha = 0.4, size = 1) +  # Create density plot with transparency
  labs(
    title = "Predicted Probability Distribution for Training Data",
    x = "Predicted Probability",
    y = "Density"
  ) +
  scale_fill_manual(values = c("blue", "red")) +  # Customize colors for each class
  scale_color_manual(values = c("blue", "red")) +  # Outline colors for the curves
  theme_minimal()






######## Steps for HoldOut datasset score.

# Holdout Data Test
holdout_data <- read.csv("ModifiedData.csv")



holdout_data <- holdout_data %>%
  select(-is_repeated_guest, -is_agent, -is_company, -distribution_channel, -babies, -is_meal, -continent)

# Step 2: Data Pre-processing
holdout_data <- holdout_data %>%
  mutate_at(c("arrival_date_year", "hotel", "market_segment", "deposit_type", "customer_type", "is_reserved_room", "arrival_season"), as.factor)


# Step 1: Create a dummy variable model from the training data
dummies_model <- dummyVars(~ ., data = hotel_train)

# Step 2: Apply the same dummy variable transformation to the holdout data
holdout_dummy <- predict(dummies_model, newdata = holdout_data)

# Step 3: Convert holdout_dummy to a data frame
holdout_dummy <- data.frame(holdout_dummy)

# Step 4: Check for missing columns and add them if necessary
missing_cols <- setdiff(colnames(predict(dummies_model, newdata = hotel_train)), colnames(holdout_dummy))

# Step 5: Add the missing columns with zero values to the holdout data
for (col in missing_cols) {
  holdout_dummy[[col]] <- 0
}

# Step 6: Reorder the columns of the holdout data to match the training data
holdout_dummy <- holdout_dummy[, colnames(predict(dummies_model, newdata = hotel_train))]


case_holdoutprob<- predict(model_xgb_tuned, holdout_dummy, type="prob")

case_holdout_scored<- cbind(holdout_dummy, case_holdoutprob$cancelled)
case_holdout_scored[1:3,]


# Save the data frame as a CSV file
write.csv(case_holdout_scored, file = "HoldoutSet.csv", row.names = FALSE)





