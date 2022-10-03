# Databricks notebook source
# MAGIC %md
# MAGIC # Data loader

# COMMAND ----------

import pandas as pd
import re
import numpy as np
from datetime import date, datetime, timedelta
pd.options.mode.chained_assignment = None

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
       

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load in data and save to wroking directory

# COMMAND ----------

cust_df = spark.read.table("hive_metastore.default.customers_data_csv")
tran_df = spark.read.table("hive_metastore.default.transactions_data_csv")

gdp = spark.read.table("hive_metastore.default.GDP_data_quarterly_csv")
income = spark.read.table("hive_metastore.default.income_data_csv")
interest = spark.read.table("hive_metastore.default.interest_rate_data_csv")
umcs = spark.read.table("hive_metastore.default.cust_senti_data_csv")
unemployment = spark.read.table("hive_metastore.default.unemployment_data_csv")
gdp = gdp.drop('GDPC1')

# COMMAND ----------

from pyspark.sql.functions import *	
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import row_number

df = tran_df.limit(100000)
display(df)

# df["volume"] = df.groupby('account_id').cumcount()
w = Window().orderBy(['account_id', "date"])
df = df.withColumn("volume", row_number().over(w))

df.withColumn("date", to_date(col("date"),"yyyy-MM-dd"))
df = df.groupby(["account_id", "date"]).agg({"account_id": "mean",
                                            "customer_id": "mean",
                                            "amount": "mean",
                                            "deposit":"mean",
                                            "withdrawal":"mean",
                                            "volume": "count"
                                            })


display(df)


# COMMAND ----------

df = df.join(cust_df, df["avg(customer_id)"] == cust_df["customer_id"])           

df['state'] = df['state'].apply(lambda x: state_to_code(x)) # todo drop australia
df = df.drop(columns=["customer_id"])
df.head()

# COMMAND ----------


df['balance'] = df.groupby('account_id')['amount'].cumsum()
df['balance'] += df['start_balance']

df['account_length'] = pd.to_datetime(df['date'])-pd.to_datetime(df['creation_date'])

df['age'] = pd.to_datetime(df['date'])-pd.to_datetime(df['dob'])

churn = df.groupby('account_id')['date'].transform('max')
df['churned'] = np.array((df["date"] == churn))

df.head()

# COMMAND ----------

gdp.index = pd.to_datetime(gdp.index, format='%Y/%m/%d')
income.index = pd.to_datetime(income.index)
interest.index = pd.to_datetime(interest.index)
umcs.index = pd.to_datetime(umcs.index)
unemployment.index = pd.to_datetime(unemployment.index)

income.dropna(inplace=True)
interest.dropna(inplace=True)
umcs.dropna(inplace=True)
unemployment.dropna(inplace=True)


# resample the GDP
gdp1 = gdp.resample('MS').ffill()
dat = gdp1.index.strftime('%Y-%d-%m')
gdp1 = gdp1.set_index(dat)

#slice dfs
gdp1 = gdp1.loc['2007-01-01':'2020-01-07']

income = income.loc['2007-01-01':]
interest = interest.loc['2007-01-01':]
umcs = umcs.loc['2007-01-01':]
unemployment = unemployment.loc['2007-01-01':]

# combine

gdp1['p_change_income'] = income['p_change_income']
gdp1['interest_rate'] = interest['interest_rate']
gdp1['UMCSENT'] = umcs['UMCSENT']
gdp1['unemp_rate'] = unemployment['unemp_rate']
gdp1['DATE'] = gdp1.index
final = gdp1
final["DATE"] = pd.to_datetime(final["DATE"], format='%Y-%d-%m')
final["DATE"] =final["DATE"].apply(lambda row: row - pd.DateOffset(days=1))
final = final.set_index('DATE') 
final.head()

# COMMAND ----------

df = df.merge(final, left_on="date", right_index=True)
df=  df.sort_values(by=['account_id', 'date'])
df.head()

# COMMAND ----------

df.to_csv("data/processed/pro_data.csv")

# COMMAND ----------

dbutils.fs.rm("/FileStore/tables/your_table_name.csv")


