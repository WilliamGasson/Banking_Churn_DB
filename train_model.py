# Databricks notebook source
# MAGIC %md
# MAGIC # train_model.py
# MAGIC 
# MAGIC Train KNN model on churn data 
# MAGIC 
# MAGIC 1) Imports
# MAGIC 2) Set up logging and random state
# MAGIC 3) Load data
# MAGIC 4) Split data
# MAGIC 5) Transform data
# MAGIC 6) Set up tracking
# MAGIC 7) Run model to find optimal value
# MAGIC 8) Find best model
# MAGIC 9) Pickle best model
# MAGIC 10) SHAP

# COMMAND ----------

__date__ = "2022-10-3"
__author__ = "WilliamGasson"
__version__ = "0.1"

# 1) Imports

import logging
import os
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, classification_report
from sklearn.model_selection import cross_val_score, cross_validate, learning_curve, train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

warnings.filterwarnings('ignore')

# 2) Logging and seed

logging.basicConfig(
    format = '[%(levelname)s] %(asctime)s - %(message)s',
    level = logging.INFO,
    datefmt = '%Y/%m/%d %H:%M:%S',
    )
logger = logging.getLogger(__name__)

seed = 123
rng = np.random.RandomState(seed)
logger.debug(f"np.random.RandomState initialised with seed {seed}")

# COMMAND ----------

# 3) Load data

df = spark.read.table(spark.read.table("hive_metastore.default.processed_data_csv"))

# dont use any data that was after 2020
df = df.drop(df[(df['date'] < 360)].index) 

X = df.drop(columns=['churned'])
y = df['churned']


# COMMAND ----------

# 4) Train/test split 

# set test data to be the data from 2019-2020
df = df.drop(df[(df['date'] < 360)].index) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rng)
logger.debug(f"X_train shape: {X_train.shape}")
logger.debug(f"y_train shape: {y_train.shape}")
logger.debug(f"X_test shape: {X_test.shape}")
logger.debug(f"y_test shape: {y_test.shape}")

logger.info("Loaded the churn dataset into train and test sets")

df_train = X_train.copy()
df_train["target"] = y_train
df_test = X_test.copy()
df_test["target"] = y_test

# Save test data for testing

if not os.path.exists('../../resources/data/df_train.csv'):
    df_train.to_csv('../../resources/data/df_train.csv', index=False)
if not os.path.exists('../../resources/data/df_test.csv'):
    df_test.to_csv('../../resources/data/df_test.csv',index=False)
    
logger.info("Saved the churn dataset train and test sets")

# Spliting for validation 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=rng)


# COMMAND ----------

# 5) Transform data 

num_col = ['amount','deposit', 'withdrawal', 'balance', 'volume',  'start_balance' /
           ,'age', 'account_length','bal_spend_ratio','bal_save_diff' /
           ,'p_change_gdp','interest_rate', 'UMCSENT','unemp_rate']
       
# Create column transformer object
col_trans = ColumnTransformer(
    [
        ('Std_Scaling', StandardScaler(),num_col),  
        ('OHE',OneHotEncoder(),['state'])
    ], remainder='drop' 
)

#Fit the ColumnTransformer to the data
Xt_train=col_trans.fit_transform(X_train)

Xt_val = col_trans.transform(X_val) 
 
# Put into dataframe
Xt_train = pd.DataFrame(Xt_train, columns= col_trans.get_feature_names_out())
Xt_val = pd.DataFrame(Xt_val, columns= col_trans.get_feature_names_out())




# COMMAND ----------

# 6) Tracking uri and mlflow experiment

# Set a folder where we record the experiment
tracking_uri = r"file:///C:/mlflow_local/mlruns"
logger.info(f"tracking uri set up to {tracking_uri}")
experiment_name = "churn type - Random forest-final"
local_dir = r"C:/Users/zackf/banking_churn/ranf_model"

# Mlflow setup - or logs will be saved in current directory
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

# From the mlflow uri tracking folder if you run mlflow ui 
# it will display the in local host

# 7) Run experiments


def eval_metrics(y_test,y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)
    roc = roc_auc_score(y_test,y_pred)
    return acc, f1, roc

max_samples_leaf = 6
max_depth = 20
n_experiments = 20
max_trees = 150
for i in range(n_experiments):
    with mlflow.start_run():
        # Get random hyperparameters
        l = rng.randint(1,max_samples_leaf)
        d = rng.randint(5,max_depth)
        t = rng.randint(100,max_trees)
        logger.info(f"Experiment {i+1}/{n_experiments}: leaf: {l}, no_trees: {t}")
        
        # Fit model
        ranf = RandomForestClassifier(min_samples_leaf=l, n_estimators=t, n_jobs=-1)#, max_depth=d, n_jobs=-1)
        scores = cross_validate(ranf, Xt_train, y_train, cv=10, scoring=['accuracy'],return_train_score=True)

        ranf.fit(Xt_train, y_train)
        y_pred = ranf.predict(Xt_val)

        print(classification_report(y_val, y_pred))

        # acc_train = accuracy_score(y_train, y_pred)
        # f1_train = f1_score(y_train,y_pred)

        # Evaluate model
        acc_train = scores['test_accuracy'].mean()
        # acc_valid = scores['train_accuracy'].mean()

        acc_val = accuracy_score(y_val, y_pred)
        f1_val = f1_score(y_val,y_pred)

        # Evaluate model
        acc_val = scores['test_accuracy'].mean()
        acc_valid = scores['train_accuracy'].mean()

        # Logging restult to mlflow
        mlflow.log_param("leaf", l)
        mlflow.log_param("depth", d)
        mlflow.log_param('trees',t)
        # mlflow.log_metric('acc_train', acc_train)
        # mlflow.log_metric('f1_train', f1_train)
        mlflow.log_metric('acc_val', acc_val)
        mlflow.log_metric('f1_val', f1_val)
        mlflow.sklearn.log_model(ranf, "model")


# COMMAND ----------

# 8) Find best run

client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
exp_id = client.get_experiment_by_name(experiment_name).experiment_id

runs = client.search_runs(exp_id, order_by=["metrics.F1 DESC"])

best_run_id = runs[0].info.run_id

print("best run:", best_run_id)


client.download_artifacts(best_run_id, "model", local_dir)

model_uri = "file:///"+os.path.join(local_dir,"model")
model = mlflow.sklearn.load_model(model_uri)

# 9) Pickle optimal model

# Pickle
dump(col_trans, '../../resources/models/standardScaler.joblib')
dump(model, '../../resources/models/ran_f.joblib')



# COMMAND ----------

# 10) Shap

shap.initjs()

f = lambda x: model.predict_proba(x)[:,1]
med = X_train.median().values.reshape((1,X_train.shape[1]))

explainer = shap.Explainer(f, med)
shap_values = explainer(X_test.iloc[0:1000,:])
shap.plots.waterfall(shap_values[0])

# %%

