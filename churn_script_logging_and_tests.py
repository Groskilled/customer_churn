'''
    Tests for functions in churn_library.py
    run pytest churn_script_logging_and_tests.py
    Date: July 2022
    Author: Adam Wybierala
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
	force=True,
    format='%(name)s - %(levelname)s - %(message)s')

def test_import():
	'''
	test data import
	'''
	try:
		df = cls.import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	df = cls.import_data("./data/bank_data.csv")
	try:
		cls.perform_eda(df)
		assert os.path.exists("./images/eda/age.png") == True
		assert os.path.exists("./images/eda/churn.png") == True
		assert os.path.exists("./images/eda/corr.png") == True
		assert os.path.exists("./images/eda/marital_status.png") == True
		assert os.path.exists("./images/eda/trans_ct.png") == True
		logging.info("Testing perform_eda: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The files weren't created")
		raise err


def test_encoder_helper():
	'''
	test encoder helper
	'''
	df = cls.import_data("./data/bank_data.csv")
	df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
	cat_columns = [
		'Gender',
		'Education_Level',
		'Marital_Status',
		'Income_Category',
		'Card_Category'
	]
	cols_to_check = [x + "_Churn" for x in cat_columns]
	try:
		df = cls.encoder_helper(df, cat_columns, 'Churn')
		for col in cols_to_check:
			assert col in df.columns
		logging.info("Testing encoder_helper: SUCCESS")
	except AssertionError as err:
		logging.error("Testing encoder_helper: can't find expected column")
		raise err



def test_perform_feature_engineering():
	'''
	test perform_feature_engineering
	'''
	df = cls.import_data("./data/bank_data.csv")
	try:
		X_train, X_test, Y_train, Y_test = cls.perform_feature_engineering(df, 'Churn')
		assert X_train.shape == (7088, 19)
		assert Y_train.shape == (7088,)
		assert X_test.shape == (3039, 19)
		assert Y_test.shape == (3039,)
		logging.info("Testing perform_feature_engineering: SUCCESS")
	except AssertionError as err:
		logging.error("Testing perform_feature_engineering: outputs have the wrong size")
		raise err


def test_train_models():
	'''
	test train_models
	'''
	df = cls.import_data("data/bank_data.csv")[:100]
	X_train, X_test, Y_train, Y_test = cls.perform_feature_engineering(df, 'Churn')
	try:
		cls.train_models(X_train, X_test, Y_train, Y_test)
		logging.info("Testing train_models: SUCCESS")
		assert os.path.exists("./images/results/importance.png") == True
		assert os.path.exists("./images/results/Logistic Regression .png") == True
		assert os.path.exists("./images/results/Random Forest .png") == True
		assert os.path.exists("./images/results/roc_curve.png") == True
		assert os.path.exists("./models/logistic_model.pkl") == True
		assert os.path.exists("./models/rfc_model.pkl") == True
	except AssertionError as err:
		logging.error("Testing train_models: file not created")
		raise err
