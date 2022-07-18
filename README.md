# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
First project of the MLOps Nanodegree.
The base code is provided by Udacity as a notebook and the data is from kaggle.
It aims to identify credit card customers that are most likely to churn using two models: an random forest classifier and a logistic regression.


## Files and data description
churn_library.py: contains the functions for data preparation, feature engineering, model training and saving the
results.

chrun_notebook.ipynb: a notebook with the starting uncleaned code.

churn_script.py: script to do everything

churn_script_logging_and_tests.py: tests for each function in churn_library.py

data/bank_data.csv: datas used for the churn prediction


## Running Files
How do you run your files? What should happen when you run your files?
To run the project first install the dependencies using pip install -r requirements_py3.X.txt

Then simply run python churn_script.py. This should create files in images/eda/ and images/results and models/logistics_model.pkl and models/rfc_model.pkl will be created.

To run the tests: pytest churn_script_logging_and_tests.py. Logs will be visible in logs/churn_library.log



