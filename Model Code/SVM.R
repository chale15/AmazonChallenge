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
library(kernlab)

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
  step_pca(all_predictors(), threshold=1)

#SVM Models

svm_radial <- svm_rbf(rbf_sigma=tune(),
                        cost=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

svm_polynomial <- svm_poly(degree=tune(),
                      cost=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

svm_lin <- svm_linear(cost=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

radial_workflow <- workflow() %>% 
  add_model(svm_radial) %>% 
  add_recipe(my_recipe)

poly_workflow <- workflow() %>% 
  add_model(svm_polynomial) %>% 
  add_recipe(my_recipe)

lin_workflow <- workflow() %>% 
  add_model(svm_lin) %>% 
  add_recipe(my_recipe)

tuning_grid_radial <- grid_regular(rbf_sigma(), cost(), levels = 5)
tuning_grid_poly <- grid_regular(degree(), cost(), levels = 5)
tuning_grid_lin <- grid_regular(cost(), levels = 5)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results_radial <- radial_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_radial, 
            metrics = metric_set(roc_auc))

cv_results_poly <- poly_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_poly, 
            metrics = metric_set(roc_auc))

cv_results_lin <- lin_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid_lin, 
            metrics = metric_set(roc_auc))

best_tune_radial <- cv_results_radial %>% select_best(metric='roc_auc')
best_tune_poly <- cv_results_poly %>% select_best(metric='roc_auc')
best_tune_lin <- cv_results_lin %>% select_best(metric='roc_auc')

final_workflow_radial <- radial_workflow %>% 
  finalize_workflow(best_tune_radial) %>% 
  fit(data = train)

final_workflow_poly <- poly_workflow %>% 
  finalize_workflow(best_tune_poly) %>% 
  fit(data = train)

final_workflow_lin <- lin_workflow %>% 
  finalize_workflow(best_tune_lin) %>% 
  fit(data = train)

radial_svm_preds <- predict(final_workflow_radial, 
                    new_data = test,
                    type = 'prob')

poly_svm_preds <- predict(final_workflow_poly, 
                            new_data = test,
                            type = 'prob')

linear_svm_preds <- predict(final_workflow_lin, 
                            new_data = test,
                            type = 'prob')

radial_svm_submission <- radial_svm_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=radial_svm_submission, file="./Submissions/RadialSVMpreds1.csv", delim=",")

poly_svm_submission <- poly_svm_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=poly_svm_submission, file="./Submissions/PolySVMpreds1.csv", delim=",")

linear_svm_submission <- linear_svm_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=linear_svm_submission, file="./Submissions/LinearSVMpreds1.csv", delim=",")
