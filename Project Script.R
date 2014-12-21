# Set Working Directory
setwd("C:/Users/Dustin/Desktop/My Dropbox/Coursera Data Science/Github/Practical Machine Learning Project Submission")

# Load required packages
sapply(c("data.table",
         "reshape2",
         "doParallel",
         "foreach",
         "caret",
         "randomForest",
         "gbm"),
       require, character.only = T)

# Code for parallel processing
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

# Read in the training and test datasets
training <- fread("Raw Data/pml-training.csv", na.strings = c("", "NA", "#DIV/0!"))
testing <- fread("Raw Data/pml-testing.csv", na.strings = c("", "NA", "#DIV/0!"))

# Check if the test and training datasets contain the same variables
identical(names(testing), names(training))
mismatchVarInd <- which(!(names(testing) %in% names(training)))
names(testing)[mismatchVarInd]
names(training)[mismatchVarInd]
identical(names(testing)[1:159], names(testing)[1:159])

# Inspect the training data
str(training) # Notice there are many NA values

# Count the number of NA values for each field
NAValues <- training[, lapply(.SD, function(x) sum(is.na(x)))]
str(NAValues) 
# Seems like many columns have at least 19216 NA Values
# Proceed to remove those columns of data since they will not be useful for prediction
NAValues <- melt(NAValues)
setkey(NAValues, "value")
NAValues <- NAValues[list(0)]
NAValues <- as.character(NAValues[["variable"]])
# Reminding ourselves that this vector will contain those variables we want to keep
# We need to add in the mismatched variable from the test dataset
NAValues <- c(NAValues, names(testing)[mismatchVarInd])
training[, names(training)[!(names(training) %in% NAValues)] := NULL]

# Remove some additional columns because they contain user data and are not helpful
names(training)[1:7]
training[, names(training)[1:7] := NULL]
str(training)

# Check for non zero variance covariates
nsv <- nearZeroVar(training, saveMetrics = T)
nsv
# Nothing to eliminate

# Convert the classe column to factor
training[, "classe" := as.factor(classe)]

# Do the same for the test set
testing[, names(testing)[!(names(testing) %in% NAValues)] := NULL]
testing[, names(testing)[1:7] := NULL]

# Search for the best 'mtry' using 10 fold cross validation
mtryGrid <- 1:52
set.seed(3)
seeds <- c(replicate(10, sample.int(1000, length(mtryGrid)), simplify = F), sample.int(1000, 1))
modelRF <- train(classe ~ ., 
                 data = training, 
                 method = "rf", 
                 trControl = trainControl(method = "cv", seeds = seeds),
                 tuneGrid = data.frame(mtry = mtryGrid))

# The time taken for the entire process
modelRF$times$everything

# Plot of Accuracy versus mtry
plot(modelRF)

# Best mtry is 9 with accuracy
mtryBest <- modelRF$bestTune$mtry
mtryBestAccuracy <- modelRF$results[9, ]
mtryBestError <- (1 - mtryBestAccuracy[1, 2])*100

# Build the final model
seeds <- as.list(sample.int(1000, 11))
modelRFFinal <- train(.outcome ~ ., 
                      data = training,
                      method = "rf",
                      trControl = trainControl(method = "cv", seeds = seeds),
                      tuneGrid = data.frame(mtry = mtryBest))
modelRFFinal$finalModel
modelRFFinal$times$everything
modelRFFinalAccuracy <- modelRFFinal$results
modelRFFinalError <- (1 - modelRFFinalAccuracy[1, 2])*100

# Apply model on test set
predictions <- predict(modelRFFinal, testing)

# Load function to write answers
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

# Write prediction answers
pml_write_files(predictions)

# Stop the cluster
stopCluster(cl)