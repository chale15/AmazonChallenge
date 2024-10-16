library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(glmnet)
library(kknn)

#Read Data

#setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/amazon-employee-access-challenge/")
setwd("~/Kaggle/AmazonChallenge")

train <- vroom("train.csv")
test <- vroom("test.csv")

train <- train %>% mutate(ACTION = as.factor(ACTION))

my_recipe <- recipe(ACTION~., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())


#Fit Model

knn_model <- nearest_neighbor(neighbors=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

knn_workflow <- workflow() %>% 
  add_model(knn_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(neighbors(), levels = 100)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- knn_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- knn_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

knn_preds <- predict(final_workflow, 
                      new_data = test,
                      type = 'prob')

#Format for Submission

knn_submission <- knn_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=knn_submission, file="./Submissions/KNNPreds2.csv", delim=",")
