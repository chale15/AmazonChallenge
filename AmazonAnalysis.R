library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)

#Read Data

setwd("~/Desktop/Fall 2024/Stat 348/GitHubRepos/amazon-employee-access-challenge/")

train <- vroom("train.csv")
test <- vroom("test.csv")


#Visualize Data

vis_recipe <- recipe(ACTION~., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor)

prep_vis <- prep(vis_recipe)
factored <- bake(prep_vis, new_data = train)

summary(factored)

corr_plot <- corrplot::corrplot(cor(train))

#Dummy Variable Encoding

my_recipe <- recipe(ACTION~., data = train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())


prepped <- prep(my_recipe)
baked <- bake(prepped, new_data= train)
