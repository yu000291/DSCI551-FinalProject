from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import pyspark.sql.functions as fc
import pandas as pd
from pyspark.sql.functions import when, col
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import json
import requests

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
#read the file
oslong_df = spark.read.option("header","true").csv("oasis_longitudinal.csv")
#drop the column
oslong_df = oslong_df.drop('Subject ID', 'MRI ID', 'Visit', 'MR Delay', 'Hand')
#rename the column
oslong_df = oslong_df.withColumnRenamed("M/F","gender")
#change gender when male to 0, when female to 1
oslong_df = oslong_df.withColumn('gender', when(col('gender') == 'M', '0').otherwise('1'))

#change to dataframe
data = oslong_df.toPandas()

# Drop the converted case
data = data[data.Group != 'Converted']
data = data.reset_index(drop=True)

# imputation on the whole dataset as there are limited amount of observations
impt = data.iloc[:,1:]

imp = IterativeImputer(max_iter=10, random_state=57)
imp_data = imp.fit_transform(impt)
impt_data = pd.DataFrame(imp_data, columns = impt.columns)
impt_data['Group'] = data['Group']

# split data into test and train dataset
train_X = impt_data.iloc[:235,:-1]
train_y = impt_data.iloc[:235,-1]

train_X = impt_data.iloc[235:,:-1]
train_y = impt_data.iloc[235:,-1]

train = impt_data.iloc[:235,:]
test = impt_data.iloc[235:,:]

train_rlt = train.to_json(orient="records")
test_rlt = test.to_json(orient="records")

url1 = 'https://finalproject-42236-default-rtdb.firebaseio.com/train.json'
url2 = 'https://finalproject-42236-default-rtdb.firebaseio.com/test.json'

r1 = requests.put(url1, train_rlt)
r2 = requests.put(url2, test_rlt)