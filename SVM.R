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

#SVM

svm_radial <- svm_rbf(rbf_sigma=tune(),
                        cost=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

radial_workflow <- workflow() %>% 
  add_model(svm_radial) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(rbf_sigma(), cost(), levels = 5)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- radial_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- radial_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

radial_svm_preds <- predict(final_workflow, 
                    new_data = test,
                    type = 'prob')

radial_svm_submission <- radial_svm_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=radial_svm_submission, file="./Submissions/RadialSVMpreds2.csv", delim=",")
