# Load required packages
sapply(c('data.table',
         'magrittr',
         'stringr',
         'doParallel',
         'foreach',
         'caret'),
       require, 
       character.only = T)

# Setup up parallel processing
cl <- 
    makeCluster(detectCores() - 1) %T>%
    registerDoParallel()

# Read in the training and test datasets
training <- fread("./Raw Data/pml-training.csv", 
                  na.strings = c("", "NA", "#DIV/0!"))
testing <- fread("./Raw Data/pml-testing.csv", 
                 na.strings = c("", "NA", "#DIV/0!"))

# The test and train datasets do not have identical variable names
# They differ on the 160th variable name
commonVariables <- intersect(names(training), names(testing))
sapply(list(training, testing),
       function(x) {
           differingVariable <- setdiff(names(x), commonVariables)
           differingVariableIndex <- grep(differingVariable, names(x))
           message(paste0('Differing variable named "', differingVariable,
                          '" at column ', differingVariableIndex, '.\n'))
       })

# Count the number of NA values for each field
# Seems like many columns have either no NA values or at least 19216 NA values
# Proceed to remove those columns of data with NA values since they will not be useful for prediction
NAValues <- 
    training[, lapply(.SD, function(x) sum(is.na(x)))] %>%
    melt() %>%
    subset(value != 0) %>%
    .[[1]] %>%
    as.character()
training[, intersect(names(training), NAValues) := NULL]

# Remove some additional columns because they contain user data and are not helpful
userDataVariables <- names(training)[1:7]
training[, intersect(names(training), userDataVariables) := NULL]

# Check for non zero variance covariates
nzv <- nearZeroVar(training, saveMetrics = T)
nzv
# Nothing to eliminate

# Convert the classe column to factor
training[, "classe" := as.factor(classe)]

# Do the same for the test set
testing[, intersect(names(testing), c(userDataVariables, NAValues)) := NULL]

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
modelRFFinal <- train(classe ~ ., 
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