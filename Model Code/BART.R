library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)
library(glmnet)
library(naivebayes)
library(ranger)
library(kknn)
library(discrim)
library(themis)
library(dbarts)
library(rpart)
library(baguette)
library(xgboost)
library(stacks)

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
  step_pca(all_predictors(), threshold=0.9)


bart_model <- parsnip::bart(trees = 1000) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification") %>% 
  translate()

bart_workflow <- workflow() %>% 
  add_model(bart_model) %>% 
  add_recipe(my_recipe)

bart_fit <- bart_workflow %>% 
  fit(data = train)

bart_preds <- bart_fit %>% predict(new_data = test, type = 'prob')

bart_submission <- bart_preds %>% 
  bind_cols(., test) %>% 
  dplyr::select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=bart_submission, file="./Submissions/BartPreds3.csv", delim=",")

## Bag Trees

bag_model <- bag_tree(min_n = tune(), cost_complexity = tune()) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

bag_workflow <- workflow() %>% 
  add_model(bag_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(min_n(), cost_complexity(), levels = 10)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- bag_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- bag_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

bag_preds <- predict(final_workflow, 
                      new_data = test,
                      type = 'prob')

bag_submission <- bag_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=bag_submission, file="./Submissions/BagPreds.csv", delim=",")

## Boosted Tree Model

bt_model <- boost_tree(mtry = tune(),
                       min_n = tune(),
                       trees = 500,
                       learn_rate = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") %>% 
  translate()

bt_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(bt_model)

bt_tuning_grid <- grid_regular(mtry(range=c(1, 9)), min_n(), learn_rate(), levels = 5)

untunedModel <- control_stack_grid() 

bt_models <- bt_workflow %>%
  tune_grid(resamples=folds,
            grid=bt_tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)

bt_bestTune <-bt_models %>% select_best(metric="roc_auc")

bt_fit <- bt_workflow %>% 
  finalize_workflow(bt_bestTune) %>% 
  fit(data = train)

bt_predict <- bt_fit %>% predict(new_data = test, type = 'prob')

bt_submission <- bt_predict %>% 
  bind_cols(., test) %>% 
  dplyr::select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=bt_submission, file="./Submissions/BoostedPreds.csv", delim=",")

