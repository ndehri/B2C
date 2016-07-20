setwd("~/myR_scripts/PredictConversionRatesOnBreakdownServices")


library(lubridate) # Dealing with Dates a Little Easier
library(dplyr)
library(tidyr)
if ("package:h2o" %in% search()) { 
        h2o.removeAll() # Clean slate - just in case the cluster was already running
        h2o.shutdown()
        detach("package:h2o", unload=TRUE) 
        }

############################
## Import the data into R
###########################

df_r <- read.csv("train.CSV") %>%
        mutate(dataset = "train") # Add a column dataset 

df_test_r <- read.csv("test.CSV") %>%
        mutate(souscrit = 0, dataset = "test") # Add a column dataset and a column souscrit

# Row binding the datasets
mydata_r <-  rbind(df_r, df_test_r)

#######################################
## Feature engineering
#######################################

#####  DT_DEBT_ASSR ##### 
# Convert DT_DEBT_ASSR (Date ou le client a ete contacte pour lui proposer DEGEX+ ou DEPEX+)
mydata_r$DT_DEBT_ASSR <- ymd(mydata_r$DT_DEBT_ASSR)

# Add a column for the previous day of DT_DEBT_ASSR
mydata_r$Prev_DT_DEBT_ASSR <-  mydata_r$DT_DEBT_ASSR - 1

# split the date DT_DEBT_ASSR
mydata_r <- mydata_r %>%
        mutate(day_DT_DEBT_ASSR = day(DT_DEBT_ASSR), month_DT_DEBT_ASSR = month(DT_DEBT_ASSR), 
               year_DT_DEBT_ASSR = year(DT_DEBT_ASSR), quarter_DT_DEBT_ASSR = quarter(DT_DEBT_ASSR), 
               weekday_DT_DEBT_ASSR = wday(DT_DEBT_ASSR))

# split the date Prev_DT_DEBT_ASSR 
mydata_r <- mydata_r %>%
        mutate(weekday_Prev_DT_DEBT_ASSR = wday(Prev_DT_DEBT_ASSR))


#####  DATE_EMMENAG ##### 
# Convert DATE_EMMENAG (Date d'emmenagement dans son logement du client)
mydata_r$DATE_EMMENAG <- ymd(mydata_r$DATE_EMMENAG)

#####  DATE_ANC_CLI ##### 
# Convert DATE_ANC_CLI (Date du premier contrat du client chez ENGIE)
mydata_r$DATE_ANC_CLI <- ymd(mydata_r$DATE_ANC_CLI)

# split the date DATE_ANC_CLI
mydata_r <- mydata_r %>%
        mutate(day_DATE_ANC_CLI = day(DATE_ANC_CLI), month_DATE_ANC_CLI = month(DATE_ANC_CLI), 
               year_DATE_ANC_CLI = year(DATE_ANC_CLI), quarter_DATE_ANC_CLI = quarter(DATE_ANC_CLI), 
               weekday_DATE_ANC_CLI = wday(DATE_ANC_CLI))

# Reconstruct the datasets
df_r <- filter(mydata_r, dataset == "train") %>% select(-dataset)
df_test_r <- filter(mydata_r, dataset == "test") %>% select(-souscrit, -dataset)

# delete mydata_r
rm(mydata_r)

#####################################
## Launch an H2O cluster on localhost
#####################################
library(h2o)
h2o.init(nthreads=-1)

############################
## Import the data into H2O
###########################

df <- as.h2o(df_r)
df_test <- as.h2o(df_test_r)

## pick a response for the supervised problem
response <- "souscrit"

## the response variable is an integer, we will turn it into a categorical/factor for binary classification
df[[response]] <- as.factor(df[[response]])           

## use all other columns (except for the response) as predictors
predictors <- setdiff(names(df), response) 


#######################################
## Split the data for Machine Learning
#######################################

splits <- h2o.splitFrame(
        data = df, 
        ratios = c(0.8),
        destination_frames = c("train.hex", "test.hex"), seed = 1234
)

train <- splits[[1]]
test <- splits[[2]]


#######################################
## Hyper-Parameter Search
#######################################

# we will use random hyper-parameter search to "let the machine get luckier than a best guess of any human"

hyper_params = list( 
        ## restrict the search of max_depth 
        max_depth = seq(1,9,1),                                      
        
        ## search a large space of row sampling rates per tree
        sample_rate = seq(0.2,1,0.01),                                             
        
        ## search a large space of column sampling rates per split
        col_sample_rate = seq(0.2,1,0.01),                                         
        
        ## search a large space of column sampling rates per tree
        col_sample_rate_per_tree = seq(0.2,1,0.01),                                
        
        ## search a large space of how column sampling per split should change as a function of the depth of the split
        col_sample_rate_change_per_level = seq(0.9,1.1,0.01),                      
        
        ## search a large space of the number of min rows in a terminal node
        min_rows = 2^seq(0,log2(nrow(train))-1,1),                                 
        
        ## search a large space of the number of bins for split-finding for continuous and integer columns
        nbins = 2^seq(4,10,1),    
        
        ## search a large space of the number of bins for split-finding for categorical columns
        nbins_cats = 2^seq(4,12,1),                                                
        
        ## search a few minimum required relative error improvement thresholds for a split to happen
        min_split_improvement = c(0,1e-8,1e-6,1e-4),                               
        
        ## try all histogram types (QuantilesGlobal and RoundRobin are good for numeric columns with outliers)
        histogram_type = c("UniformAdaptive","QuantilesGlobal","RoundRobin")       
)

search_criteria = list(
        ## Random grid search
        strategy = "RandomDiscrete",      
        
        ## limit the runtime to 12 hours
        max_runtime_secs = 43200,         
        
        ## build no more than 500 models
        max_models = 500,                  
        
        ## random number generator seed to make sampling of parameter combinations reproducible
        seed = 1234,                        
        
        ## early stopping once the leaderboard of the top 10 models is converged to 0.01% relative difference
        stopping_rounds = 10,                
        stopping_metric = "AUC",
        stopping_tolerance = 1e-4
)

grid <- h2o.grid(
        ## hyper parameters
        hyper_params = hyper_params,
        
        ## hyper-parameter search configuration (see above)
        search_criteria = search_criteria,
        
        ## which algorithm to run
        algorithm = "gbm",
        
        ## identifier for the grid, to later retrieve it
        grid_id = "final_grid", 
        
        ## standard model parameters
        x = predictors, 
        y = response, 
        training_frame = train, 
        validation_frame = test,
        
        ## more trees is better if the learning rate is small enough
        ## use "more than enough" trees - we have early stopping
        ntrees = 10000,                                                            
        
        ## smaller learning rate is better
        ## since we have learning_rate_annealing, we can afford to start with a bigger learning rate
        learn_rate = 0.1,                                                         
        
        ## learning rate annealing: learning_rate shrinks by 1% after every tree 
        ## (use 1.00 to disable, but then lower the learning_rate)
        learn_rate_annealing = 0.99,     
        ## early stopping based on timeout (no model should take more than 1 hour - modify as needed)
        max_runtime_secs = 3600,                                                 
        
        ## early stopping once the validation AUC doesn't improve by at least 0.01% for 5 consecutive scoring events
        stopping_rounds = 5, stopping_tolerance = 1e-4, stopping_metric = "AUC", 
        
        ## score every 10 trees to make early stopping reproducible (it depends on the scoring interval)
        score_tree_interval = 10,                                                
        
        ## base random number generator seed for each model (automatically gets incremented internally for each model)
        seed = 1234                                                             
)

## Sort the grid models by AUC
sortedGrid <- h2o.getGrid("final_grid", sort_by = "auc", decreasing = TRUE)    

## We can inspect the best 10 gbm models from the grid search explicitly, and query their validation AUC
for (i in 1:10) {
        gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
        print(h2o.auc(h2o.performance(gbm, valid = TRUE)))
}

## Features importance of the top 10 gbm models
for (i in 1:10) {
        gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
        imp <- h2o.varimp(gbm)
        imp <- data.frame(imp)
        write.csv(imp, paste("../Variable_Importance_gbm",i,".csv", sep=""), row.names = FALSE)
}

#########################################################
## Cross validating and Averaging the top 10 models
#########################################################

prob = NULL
k=10
for (i in 1:k) {
        gbm <- h2o.getModel(sortedGrid@model_ids[[i]])
        cvgbm <- do.call(h2o.gbm,
                         ## update parameters in place
                         {
                                 p <- gbm@parameters
                                 p$model_id = NULL          ## do not overwrite the original grid model
                                 p$training_frame = df      ## use the full dataset
                                 p$validation_frame = NULL  ## no validation frame
                                 p$nfolds = 5               ## cross-validation
                                 p
                         }
        )
        if (is.null(prob)) prob = h2o.predict(cvgbm, df_test)$p1
        else prob = prob + h2o.predict(cvgbm, df_test)$p1
        
        ## Features importance of the 10 gbm models cross validated
        imp <- h2o.varimp(cvgbm)
        imp <- data.frame(imp)
        write.csv(imp, paste("../Variable_Importance_cvgbm",i,".csv", sep=""), row.names = FALSE)
}
prob <- prob/k
prob <- as.data.frame(prob)
names(prob) <- "predictions"
write.csv(prob,"../submission.csv",row.names = FALSE)
