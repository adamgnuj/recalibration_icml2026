# script.R


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("Usage: Rscript drf_train_predict.R X_train.csv y_train.csv X_test.csv y_test_pred.csv")
}

X_train_path <- args[1]
y_train_path <- args[2]
X_test_path  <- args[3]
X_val_path  <- args[4]
y_pred_test_path  <- args[5]
y_pred_val_path  <- args[6]
y_pred_train_path  <- args[7]

# print all imputed paths:  
# print(paste("X_train_path:", X_train_path))
# print(paste("y_train_path:", y_train_path))
# print(paste("X_test_path:", X_test_path))
# print(paste("X_val_path:", X_val_path))
# print(paste("y_pred_test_path:", y_pred_test_path))
# print(paste("y_pred_val_path:", y_pred_val_path))
# print(paste("y_pred_train_path:", y_pred_train_path))

require(drf)

# Read data
X_train <- as.matrix(read.csv(X_train_path, header=TRUE))
y_train <- as.matrix(read.csv(y_train_path, header=TRUE))
X_test  <- as.matrix(read.csv(X_test_path, header=TRUE))
X_valid  <- as.matrix(read.csv(X_val_path, header=TRUE))

# Fit DRF model
set.seed(123)
fit <- drf(X = X_train, Y = y_train, num.trees = 2000, splitting.rule = "FourierMMD")

# Predict response for X_test and X_valid
y_test_pred <- predict(fit, newdata = X_test)
y_val_pred <- predict(fit, newdata = X_valid)
y_train_pred <- predict(fit, newdata = X_train)

# Save predictions to CSV
write.csv(as.matrix(y_test_pred$weights), file = y_pred_test_path, row.names = FALSE)
write.csv(as.matrix(y_val_pred$weights), file = y_pred_val_path, row.names = FALSE)
write.csv(as.matrix(y_train_pred$weights), file = y_pred_train_path, row.names = FALSE)
