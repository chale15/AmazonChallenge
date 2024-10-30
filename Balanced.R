library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(glmnet)
library(ranger)

#Read Data

#setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/amazon-employee-access-challenge/")
setwd("~/Kaggle/AmazonChallenge")

train <- vroom("train.csv")
test <- vroom("test.csv")

train <- train %>% mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION~., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_predictors(), threshold=1) %>% 
  step_smote(all_outcomes(), neighbors=3)


#Fit RF Model

balanced_rf <- rand_forest(mtry=tune(),
                        min_n=tune(),
                        trees=500) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

balanced_workflow <- workflow() %>% 
  add_model(balanced_rf) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(mtry(range=c(1, 9)), min_n(), levels = 3)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- balanced_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- balanced_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

balanced_preds <- predict(final_workflow, 
                    new_data = test,
                    type = 'prob')


#Format for Submission

balanced_submission <- balanced_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=balanced_submission, file="./Submissions/BalancedPredsRF1.csv", delim=",")
