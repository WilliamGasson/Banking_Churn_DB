# Databricks notebook source
# MAGIC %md
# MAGIC # Data loader

# COMMAND ----------

import pandas as pd
import re

from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import row_number, udf, col, sys, datediff, to_date, mean, stddev
from pyspark.sql.types import StringType

# COMMAND ----------

def state_to_code( x):
    """
    Turns states to state code
    
    :param x : The string in question
    :returns: prediction of the intended state code
    """
    states = {'AK': 'Alaska','AL': 'Alabama','AR': 'Arkansas','AS': 'American Samoa','AZ': 'Arizona',
                'CA': 'California', 'CO': 'Colorado','CT': 'Connecticut','DC': 'District of Columbia',
                'DE': 'Delaware','FL': 'Florida','GA': 'Georgia','GU': 'Guam','HI': 'Hawaii','IA': 'Iowa',
                'ID': 'Idaho','IL': 'Illinois','IN': 'Indiana','KS': 'Kansas','KY': 'Kentucky','LA': 'Louisiana',
                'MA': 'Massachusetts','MD': 'Maryland','ME': 'Maine','MI': 'Michigan','MN': 'Minnesota',
                'MO': 'Missouri','MP': 'Northern Mariana Islands','MS': 'Mississippi','MT': 'Montana',
                'NC': 'North Carolina','ND': 'North Dakota','NE': 'Nebraska','NH': 'New ','NJ': 'New Jersey',
                'NM': 'New Mexico','NV': 'Nevada','NY': 'New York','OH': 'Ohio','OK': 'Oklahoma','OR': 'Oregon',
                'PA': 'Pennsylvania','PR': 'Puerto Rico','RI': 'Rhode Island','SC': 'South Carolina',
                'SD': 'South Dakota','TN': 'Tennessee','TX': 'Texas','UT': 'Utah','VA': 'Virginia',
                'VI': 'Virgin Islands','VT': 'Vermont','WA': 'Washington','WI': 'Wisconsin','WV': 'West Virginia',
                'WY': 'Wyoming'}
    
    if len(x) == 2: # Try another way for 2-letter codes
        for a,n in states.items():
            if len(n.split()) == 2:
                if "".join([c[0] for c in n.split()]).lower() == x.lower():
                    return a.upper()
    new_rx = re.compile(r"\w*".join([ch for ch in x]), re.I)
    for a,n in states.items():
        if new_rx.match(n):
            return a.upper()

state_udf = udf(state_to_code, StringType()) #

def remove_outliers(df,columns,n_std):
    """
    Function to remove any value past a number of standard deviations from the mean of values in a dateframe
    
    :param df: The dateframe you want to remove the outliers from
    :param columns: The name of the columns
    :returns: Returns dataframe with outliers removed
    """
    
    for column in columns:
        avg = df.agg({column: 'mean'})
        av = avg.first()[f'avg({column})']
        std = df.agg({column: 'stddev'})
        sd = std.first()[f'stddev({column})']

        df = df[(df[column] <= av + (n_std * sd))]
        df = df[(df[column] >= av - (n_std * sd))]
    return df



# COMMAND ----------

# Load data
cust_df = spark.read.table("hive_metastore.default.customers_data_csv")
tran_df = spark.read.table("hive_metastore.default.transactions_data_csv")

# remove outliers and null
cust_df = cust_df.dropna()
tran_df =tran_df.dropna()
cust_df = remove_outliers(cust_df, ['start_balance'], 4)         
tran_df = remove_outliers(tran_df, ['deposit', 'amount', 'withdrawal'], 4)  

display(tran_df)


# COMMAND ----------

df = tran_df

# Add a column for volume
w = Window().orderBy(['account_id', "date"])
df = df.withColumn("volume", row_number().over(w))

# Aggragate to monthly data
df = df.groupby(["account_id", "date"]).agg({"account_id": "mean",
                                            "customer_id": "mean",
                                            "amount": "sum",
                                            "deposit":"sum",
                                            "withdrawal":"sum",
                                            "volume": "count"
                                            })

df = df.withColumnRenamed("avg(customer_id)", "customer_id") \
       .withColumnRenamed("sum(amount)","amount")\
       .withColumnRenamed("sum(withdrawal)","withdrawal")\
       .withColumnRenamed("sum(deposit)","deposit")\
       .withColumnRenamed("count(volume)","volume") \
       .drop("avg(account_id)") \
       .withColumn("customer_id", col("customer_id").cast("double"))

# Join customer infromation to transactions
df = df.join(cust_df,"customer_id")  
df = df.drop("customer_id")

df= df.withColumn("state", state_udf("state")) #

# Add columns
df = df.withColumn('balance', F.sum(df["amount"]).over(Window.partitionBy('account_id').orderBy("date").rowsBetween(-sys.maxsize, 0)))
df = df.withColumn("balance", col("balance")+col("start_balance"))
df= df.withColumn('account_length',datediff(col("date"),col("creation_date")))
df= df.withColumn('age',datediff(col("date"),col("dob")))
df = df.drop("dob", "creation_date")
# Calculate churn
w2 = Window.partitionBy("account_id")
df = df.withColumn("last_date", F.max("date").over(w2))
df = df.withColumn('isChurn', F.when((F.col("last_date") == col('date')),1).otherwise(0))
df = df.drop("last_date")
display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Macro Economics

# COMMAND ----------

gdp = spark.read.table("hive_metastore.default.GDP_data_quarterly_csv")

# resample the GDP in pandas
gdp = gdp.toPandas()
gdp["date"] = pd.to_datetime(gdp["date"])
gdp = gdp.set_index('date').resample('M').ffill()
gdp = gdp.loc['2007-01-31':'2020-05-31']
gdp = gdp.reset_index()
gdp_rs = spark.createDataFrame(gdp) 


income = spark.read.table("hive_metastore.default.income_data_csv")
interest = spark.read.table("hive_metastore.default.interest_rate_data_csv")
umcs = spark.read.table("hive_metastore.default.cust_senti_data_csv")
unemployment = spark.read.table("hive_metastore.default.unemployment_data_csv")

macro_df = interest.join(income, "date","inner")
macro_df = macro_df.join(umcs,"date","inner")
macro_df = macro_df.join(unemployment,"date","inner")
macro_df = macro_df.withColumn("date",to_date(col("date"),"dd/MM/yyyy")) 

macro_df = macro_df.join(gdp_rs,"date","inner")

display(macro_df)


# COMMAND ----------

# Combine macro with transction data
df = df.join(macro_df,"date", "inner")
display(df)

# COMMAND ----------

# Create an unseen dataset
# drop most recent data
df = df.withColumn("age", col("age").cast("double")) \
    .withColumn("account_length", col("account_length").cast("double")) \
    .withColumn("volume", col("volume").cast("double")) 

df = df.filter(F.col("date") < F.unix_timestamp(F.lit('2020-01-01 00:00:00')).cast('timestamp'))
# test is between 2018- 2020
df_test = df.filter(F.col("date") > F.unix_timestamp(F.lit('2018-01-01 00:00:00')).cast('timestamp'))
# train is between 2007- 2018
df_train = df.filter(F.col("date") < F.unix_timestamp(F.lit('2018-01-01 00:00:00')).cast('timestamp'))



# COMMAND ----------

# save data
dbutils.fs.rm('churn_data_csv')
df_test.write.saveAsTable("churn_test")
df_train.write.saveAsTable("churn_train")

# COMMAND ----------



# COMMAND ----------


