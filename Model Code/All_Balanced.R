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
  step_pca(all_predictors(), threshold=0.9) %>% 
  step_smote(all_outcomes(), neighbors=3)



#Logistic Regression Model

log_reg_model <- logistic_reg() %>% 
  set_engine('glm')

log_reg_workflow <- workflow() %>% 
  add_model(log_reg_model) %>% 
  add_recipe(my_recipe) %>% 
  fit(data = train)

log_reg_preds <- predict(log_reg_workflow, 
                         new_data = test,
                         type = 'prob')

log_reg_submission <- log_reg_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=log_reg_submission, file="./Submissions/LogRegPreds7.csv", delim=",")
##.86111


#Penalized Logistic Model

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

plog_submission <- plog_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=plog_submission, file="./Submissions/PLogPreds5.csv", delim=",")
##.87023

#KNN Model

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

knn_submission <- knn_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=knn_submission, file="./Submissions/KNNPreds5.csv", delim=",")
##.74584

#Naive Bayes Model

nb_model <- naive_Bayes(Laplace=tune(),
                        smoothness=tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_workflow <- workflow() %>% 
  add_model(nb_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(Laplace(), smoothness(), levels = 5)

folds <- vfold_cv(train, v = 10, repeats=1)

cv_results <- nb_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- nb_workflow %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train)

nb_preds <- predict(final_workflow, 
                    new_data = test,
                    type = 'prob')

nb_submission <- nb_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=nb_submission, file="./Submissions/NBPreds5.csv", delim=",")
# .85432

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

balanced_submission <- balanced_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=balanced_submission, file="./Submissions/BalancedPredsRF1.csv", delim=",")
## .85346

