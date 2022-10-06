# Databricks notebook source
# MAGIC %md
# MAGIC # train_dt_model_ps.py
# MAGIC 
# MAGIC Train KNN model on churn data 
# MAGIC 
# MAGIC 1) Load and Split the data
# MAGIC 2) Standardise data
# MAGIC 3) Run model
# MAGIC 4) Look at feature importance
# MAGIC 5) Evaluate accuracy

# COMMAND ----------

# load and split data
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline

# load data
df = spark.read.table("default.churn_train")


# df.toPandas()
df = df.drop("date", "account_id") \
    .withColumn("age", col("age").cast("double")) \
    .withColumn("account_length", col("account_length").cast("double")) \
    .withColumn("volume", col("volume").cast("double")) 

# #split data
train_df, test_df = df.randomSplit([.8, .2], seed=42)



# COMMAND ----------

# baseline model

# catagorical data (state)
# index_output_cols = "stateIndex"
# string_indexer = StringIndexer(inputCols=["state"], outputCols=["stateIndex"], handleInvalid="skip")

categorical_cols = [field for (field, dataType) in train_df.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")

# numeric columns
numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "isChurn"))]
assembler_inputs = index_output_cols + numeric_cols

# just use numeric
vec_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")


# # baseline model decision tree
dt = DecisionTreeClassifier(labelCol="isChurn")
stages = [string_indexer, vec_assembler, dt]
pipeline = Pipeline(stages=stages)
dt.setMaxBins(40)
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

import pandas as pd

dt_model = pipeline_model.stages[-1]
display(dt_model)

dt_model.featureImportances

features_df = pd.DataFrame(list(zip(vec_assembler.getInputCols(), dt_model.featureImportances)), columns=["feature", "importance"])
features_df
dbutils.widgets.text("top_k", "5")
top_k = int(dbutils.widgets.get("top_k"))

top_features = features_df.sort_values(["importance"], ascending=False)[:top_k]["feature"].values
print(top_features)

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

display(pred_df.select("features", "isChurn", "prediction").orderBy("isChurn", ascending=False))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Evaluate model

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol="isChurn", metricName="areaUnderROC")

ROC_AUC = evaluator.evaluate(pred_df)
print(f"Area Under ROC is {ROC_AUC}")

#f1" (default), "precision", "recall", "weightedPrecision", "weightedRecall"


# COMMAND ----------


