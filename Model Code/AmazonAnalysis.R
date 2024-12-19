library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(workflows)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/amazon-employee-access-challenge/")

train <- vroom("train.csv")
test <- vroom("test.csv")

train <- train %>% mutate(ACTION = as.factor(ACTION))


#Visualize Data

#vis_recipe <- recipe(ACTION~., data = train) %>% 
#  step_mutate_at(all_numeric_predictors(), fn = factor)

#prep_vis <- prep(vis_recipe)
#factored <- bake(prep_vis, new_data = train)

#summary(factored)

#corr_plot <- corrplot::corrplot(cor(train))

#Dummy Variable Encoding

my_recipe <- recipe(ACTION~., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())


#prepped <- prep(my_recipe)
#baked <- bake(prepped, new_data= train)

#Fit Model

log_reg_model <- logistic_reg() %>% 
  set_engine('glm')

log_reg_workflow <- workflow() %>% 
  add_model(log_reg_model) %>% 
  add_recipe(my_recipe) %>% 
  fit(data = train)

log_reg_preds <- predict(log_reg_workflow, 
                         new_data = test,
                         type = 'prob')

#Format for Submission

log_reg_submission <- log_reg_preds %>% 
  bind_cols(., test) %>% 
  select(id, .pred_1) %>% 
  rename(ACTION = .pred_1) 

vroom_write(x=log_reg_submission, file="./LogRegPreds1.csv", delim=",")

