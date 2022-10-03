# Databricks notebook source
# MAGIC %md
# MAGIC # Create table
# MAGIC Code to turn csv files from parquet in file store to tables

# COMMAND ----------

# File location and type
file_location_cust = "/FileStore/tables/customers_data.csv"
file_location_tran = "/FileStore/tables/transactions_data.csv"

file_location_sent = "/FileStore/tables/cust_senti_data.csv"
file_location_gdpq = "/FileStore/tables/GDP_data_quarterly.csv"
file_location_inco = "/FileStore/tables/income_data.csv"
file_location_inte = "/FileStore/tables/interest_rate_data.csv"
file_location_unem = "/FileStore/tables/unemployment_data.csv"

file_type = "parquet"


# CSV options
infer_schema = "True"
first_row_is_header = "True"
delimiter = ","



# COMMAND ----------

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_cust)

display(df)

# COMMAND ----------

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_tran)

display(df)

# COMMAND ----------

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_sent)

display(df)

# COMMAND ----------

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_gdpq)

display(df)

# COMMAND ----------

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_inco)

display(df)

# COMMAND ----------

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location_unem)

display(df)
