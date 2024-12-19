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
library(MASS)


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
  step_pca(all_predictors(), threshold=0.9) #%>% 
#step_smote(all_outcomes(), neighbors=3)

## Linear Discrim

lin_discrim_model <- discrim_linear() %>% 
  set_engine("MASS") %>% 
  set_mode("classification")

lin_discrim_workflow <- workflow() %>% 
  add_model(lin_discrim_model) %>% 
  add_recipe(my_recipe)

tuning_grid <- grid_regular(penalty(), levels = 10)

folds <- vfold_cv(train, v = 10, repeats=1)

#cv_results <- lin_discrim_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

#best_tune <- cv_results %>% select_best(metric='roc_auc')

final_workflow <- lin_discrim_workflow %>% 
 # finalize_workflow(best_tune) %>% 
  fit(data = train)

lin_discrim_preds <- final_workflow %>% predict(new_data = test, type = 'prob')

l_d_submission <- lin_discrim_preds %>% 
  bind_cols(., test) %>% 
  rename(ACTION = .pred_1) %>% 
  dplyr::select(id, ACTION) 

vroom_write(x=l_d_submission, file="./Submissions/LinDiscrimPreds.csv", delim=",")

