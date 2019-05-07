#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This is an test exercise for Bank Hapoalim DE position

#local mac work around - add pyspark to sys.path at runtime - MAY NOT BE REQUIRED ON OTHER SYSTEMS
import findspark
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.conf import SparkConf

spark = SparkSession.builder.master("local").appName('bank-hapoalim').getOrCreate()
file = "dataset_31_credit-g.csv"
path = "data/mllib/"
filePath = path + file
#print filePath
dataFile = spark.sparkContext.textFile(filePath) # "dataset_31_credit-g.csv")

header = dataFile.first() #column headings
schemaString = header.replace('"','')  # get rid of the double-quotes
#print schemaString  # inspect clean col. names
#for convinience read all as strings, to later specify types as needed
fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split(',')]
#by inspection, manually change data types as needed
fields[1].dataType = IntegerType()  # duration
fields[4].dataType = IntegerType()  # credit_amount
fields[7].dataType = IntegerType()  # installment_commitment
fields[10].dataType = IntegerType() # residence_since
fields[12].dataType = IntegerType() # age
fields[15].dataType = IntegerType() # existing_credits
fields[17].dataType = IntegerType() # num_dependents

schema = StructType(fields)
#get rid  of header
dataFileNoHead = dataFile.filter(lambda x: x != header)
#print(dataFileNoHead.take(5)) #visual on data
#map and get rid of single quotes in string fields
mapped_data = dataFileNoHead.map(lambda k: k.split(",")).map(lambda p: (p[0].replace("'",""), int(p[1]), p[2].replace("'",""), p[3].replace("'",""), int(p[4]) , p[5].replace("'",""), p[6].replace("'","") , int(p[7]), p[8].replace("'",""), p[9].replace("'",""), int(p[10]), p[11].replace("'",""), int(p[12]), p[13].replace("'",""), p[14].replace("'",""), int(p[15]), p[16].replace("'",""), int(p[17]), p[18].replace("'",""), p[19].replace("'",""), p[20].replace("'","")))
# create the DataFrame register the DataFrame as a table
data_df = spark.createDataFrame(mapped_data, schema).cache()
data_df.createOrReplaceTempView("borrowers")
print "\n" *5 # clear screen
print("------- Bank Hapoalim Exercise Output -------")
# 2a. distinct values in 'credit_history' column (should be third col)
print("--- Ex. 2a. distinct values in credit_history column ")
distinctValuesDF = data_df.select('credit_history').distinct().show()
print '--- Ex. 2b. credit request >10,000 by unemployed male at least 30 y.o.'
print '--> only displaying purpose column for brevity'
# SQL can be run over DataFrames that have been registered as a table.
ex2b = spark.sql("SELECT purpose FROM borrowers WHERE credit_amount > 10000 AND employment == 'unemployed' AND personal_status LIKE 'male%' AND age>=30")
ex2b.show()
print '--- Ex. 2c. Average requested amount grouped by purpose, descending'
ex2c = spark.sql("SELECT purpose, ROUND(AVG(credit_amount)) as AvgAmt FROM borrowers GROUP BY purpose ORDER BY AvgAmt DESC")
ex2c.show()
print '--- Ex. 2d. BONUS - Number of customers employed over 4 yrs.'
# distinct = data_df.select('employment').distinct().show()
# '>=7', '4<=X<7'
ex2d = spark.sql("SELECT COUNT(*) as Distinguished FROM borrowers WHERE employment == '>=7' OR employment == '4<=X<7' ")
ex2d.show()
print '--- Ex. 3 Predictive Model Classifying Customer as a Good or Bad credit risk'

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
# ref. https://spark.apache.org/docs/latest/ml-features.html

# XXXXXXXXXXXXXXXXXXXXX StringIndexer - turns categorical text into index
from pyspark.ml.feature import StringIndexer
print 'Index the categorical columns to numeric representation using StringIndexer ...\n'
#rename dataframe for convinience
df = data_df

from pyspark.ml import Pipeline
#find columns of string type
dfTypes = [f.dataType for f in df.schema.fields] # get col Types
# get indeces of columns with int types to exclude from being indexed by StringIndexer()
intIndx = [i for i,x in enumerate(dfTypes) if str(x) == 'IntegerType']
listOfIntTypeCol = []          ## Start as the empty list and build list of columns that are int as typed
for i in intIndx:
  listOfIntTypeCol.append(df[i])
# separate out by col type
intTypedColDf = df.select(listOfIntTypeCol)
strCatCols = list(set(df.columns) - set(intTypedColDf.columns))
strTypedCatColDf = df.select(strCatCols)

# run StringIndexer() on categorical string type columns
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(strTypedCatColDf) for column in list(set(strTypedCatColDf.columns)) ]
# build as pipeline and transform
pipeline = Pipeline(stages=indexers)
#df.toPandas().to_csv('dfB4.csv')
df = pipeline.fit(strTypedCatColDf).transform(df) #this can change order on columns
#df.toPandas().to_csv('dfAFTER.csv') #good to here
# build list of columns ending with '_index'
suffix = '_index'
indexList = [i for i,x in enumerate(df.columns) if x.endswith(suffix)] # indeces of indexed col.
indexedColumnList = [] # build list of columns
for c in indexList: #categorical columns
    if 'class_index' in str(df[c]):
        classCol = df[c]
    else:
        indexedColumnList.append(df[c]) # add index col. except class_index which want at end
desiredColumnList = listOfIntTypeCol + indexedColumnList # combined list: numerical & categorical
# put class_index column at right most IAW math convention
desiredColumnList.append(classCol)
df_mundged = df.select(desiredColumnList)
df_mundged.toPandas().to_csv('df_mundged.csv')


#  XXXXXX Vector assembler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
dataset = df_mundged
# numerical
assembler = VectorAssembler(
    inputCols=["duration", "credit_amount",	"installment_commitment",
               "residence_since", "age", "existing_credits", "num_dependents"],
    outputCol="num_feat")
numOutput = assembler.transform(dataset)
# categorical
assembler = VectorAssembler(
    inputCols=["existing_credits",	"num_dependents",	"job_index",
    	       "foreign_worker_index",	"other_payment_plans_index",
               "savings_status_index",	"employment_index",	"housing_index",
               "personal_status_index", "own_telephone_index",	"checking_status_index",
               "purpose_index", "property_magnitude_index", "credit_history_index",
               "other_parties_index"],
    outputCol="cat_feat")
catOutput = assembler.transform(dataset)
numOutput.select("num_feat").show(truncate=False)
catOutput.select("cat_feat").show(truncate=False)

'''

# XXXXXX PCA Good for numeric data dimentionality reduction
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

df = df_mundged.select(df_mundged.columns[:6])
df.show()
pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)


# XXXXXXXXXXXXXXXXXXXXX ChiSquared Feature Selector - categorical features
#The Chi-Square test of independence is a statistical test to determine if there is a significant relationship between 2 categorical variables
#rules to use the Chi-Square Test:
#1. Variables are Categorical
#2. Frequency is at least 5
#3. Variables are sampled independently
#
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

df = spark.createDataFrame([
    (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
    (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
    (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)], ["id", "features", "clicked"])

selector = ChiSqSelector(numTopFeatures=1, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="clicked")

result = selector.fit(df).transform(df)

print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
result.show()
#XXXXXXXXXXXXXXXXXXXXXX
# XXXXXXXXXXXXXXXXXXXXX  Train-Validation split
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# Prepare training and test data.
data = spark.read.format("libsvm")\
    .load("data/mllib/sample_linear_regression_data.txt")
train, test = data.randomSplit([0.9, 0.1], seed=12345)

lr = LinearRegression(maxIter=10)

# We use a ParamGridBuilder to construct a grid of parameters to search over.
# TrainValidationSplit will try all combinations of values and determine best model using
# the evaluator.
paramGrid = ParamGridBuilder()\
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
    .build()

# In this case the estimator is simply the linear regression.
# A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=RegressionEvaluator(),
                           # 80% of the data will be used for training, 20% for validation.
                           trainRatio=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
model = tvs.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
model.transform(test)\
    .select("features", "label", "prediction")\
    .show()

# XXXXXXXXXXXXXXXXXXXXX LogisticRegression
from pyspark.ml.classification import LogisticRegression

# Load training data
training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

# Fit the model
lrModel = lr.fit(training)

# Print the coefficients and intercept for logistic regression
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

# We can also use the multinomial family for binary classification
mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8, family="multinomial")

# Fit the model
mlrModel = mlr.fit(training)

# Print the coefficients and intercepts for logistic regression with multinomial family
print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
print("Multinomial intercepts: " + str(mlrModel.interceptVector))


# XXXXXXXXXXXXXXXXXXXXX contd metrics on LR

from pyspark.ml.classification import LogisticRegression

# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = lrModel.summary

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show()
print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# Set the model threshold to maximize F-Measure
fMeasure = trainingSummary.fMeasureByThreshold
maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
    .select('threshold').head()['threshold']
lr.setThreshold(bestThreshold)



# XXXXXXXXXXXXXXXXXXXXXX IndexToString - get back category labels after processing

from pyspark.ml.feature import IndexToString, StringIndexer

df = spark.createDataFrame(
    [(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c")],
    ["id", "category"])

indexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = indexer.fit(df)
indexed = model.transform(df)

print("Transformed string column '%s' to indexed column '%s'"
      % (indexer.getInputCol(), indexer.getOutputCol()))
indexed.show()

print("StringIndexer will store labels in output column metadata\n")

converter = IndexToString(inputCol="categoryIndex", outputCol="originalCategory")
converted = converter.transform(indexed)

print("Transformed indexed column '%s' back to original string column '%s' using "
      "labels in metadata" % (converter.getInputCol(), converter.getOutputCol()))
converted.select("id", "categoryIndex", "originalCategory").show()

'''


spark.stop()
