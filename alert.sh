#!/bin/sh

R CMD BATCH --no-save --no-restore ./KNN.R &

echo "Job Completed" | mail -s "Server Update" chale15@byu.edu
