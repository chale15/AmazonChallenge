library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(glmnet)

#Read Data

#setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/amazon-employee-access-challenge/")
setwd("~/Kaggle/AmazonChallenge")

train <- vroom("train.csv")
test <- vroom("test.csv")

train <- train %>% mutate(ACTION = as.factor(ACTION))

my_recipe_log <- recipe(ACTION~., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())

log_reg_model <- logistic_reg() %>% 
  set_engine('glm')

log_reg_workflow <- workflow() %>% 
  add_model(log_reg_model) %>% 
  add_recipe(my_recipe_log) %>% 
  fit(data = train)

log_reg_preds <- predict(log_reg_workflow, 
                         new_data = test,
                         type = 'prob')

#Format for Submission

log_reg_submission <- log_reg_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=log_reg_submission, file="./LogRegPreds2.csv", delim=",")

#Dummy Variable Encoding

my_recipe <- recipe(ACTION~., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())


#Fit Model

plog_model <- logistic_reg(mixture=tune(), penalty = tune()) %>% 
  set_engine('glmnet')

plog_workflow <- workflow() %>% 
  add_model(plog_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(penalty(), mixture(), levels = 100)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- plog_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- plog_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

plog_preds <- predict(final_workflow, 
                         new_data = test,
                         type = 'prob')

#Format for Submission

plog_submission <- plog_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=plog_submission, file="./Submissions/PLogPreds2.csv", delim=",")
