'''
    This is a script that exacutes all project from command line.
    It will save the images and models which can be later viewed
    Date: July 2022
    Author: Adam Wybierala
'''

from churn_library import import_data, perform_eda, perform_feature_engineering, train_models

df = import_data("data/bank_data.csv")
perform_eda(df)
X_train, X_test, Y_train, Y_test = perform_feature_engineering(df, 'Churn')
train_models(X_train, X_test, Y_train, Y_test)
