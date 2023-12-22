# Covid 19 Analysis With Hadoop and Pyspark

## The First Step is Loading Data into Hadoop and Starting Session in Pyspark

## 1. Ingesting data in hadoop
 To ingest file in hadoop using this code on terminal 
 ```bash
     hdfs dfs -copyFromLocal /path/to/dataset.csv /dataset.csv
 ```
 To remove this file this is the code used 
 ```bash
    hdfs dfs -rm /dataset.csv
 ```

![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/1.ingestion.png)


### 1.1 Starting Session in Pyspark

This will intiate a session connection with hadoop on the root directory via port 9000

```python
  spark = SparkSession.builder \
    .appName("PySparkHDFSExtractor")\
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.jars", "hdfs://localhost:9000/share/hadoop/common/hadoop-common.jar") \
    .getOrCreate()
```
This allow the code to be extracted afterwards and be used in Pyspark for analysis 

![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/2.spark%20session%20builder.png)

```python
csv_path = "hdfs://localhost:9000/kencovid.csv"
df = spark.read.option("header","true").csv(csv_path)
```
![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/3.reading%20csv%20file.png)



## 2 Analyzing/Pre processing data 

```python
from pyspark.sql.functions import *
df.groupBy("County").count().orderBy("count", ascending=False).show()
```

![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/6.preprocess.png)

### 2.1 Filering data by handling mussing values

By counting the number of null values in each column of a pyspark Dataframe 

```python
df = df.na.fill(0)
null_counts = [df.filter(df[c].isNull()).count() for c in df.columns]
```

![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/8.filter%20outlier%20values.png)

### 2.2 Converting the columns into interger data type (integer)

```python
from pyspark.sql.functions import col

df = df.withColumn("Regular Isolation Beds Available", col("Regular Isolation Beds Available").cast("integer"))
df = df.withColumn("County", col("County").cast("string"))
```

![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/7.convert%20to%20data%20type.png)


### 2.3 Removing incorrect values 

```python
from pyspark.sql.functions import col

df_cleaned = df.where(col("Regular Isolation Beds Available").between(min_val, max_val))

```
![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/5.show%20rows.png)

After applying all the transformations and consolidated the appropriate data that is need for Training the model 
We can display the data using this code

```python
df.printSchema()
df.show(n=5)
```

![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/9.display%20dataframe.png)

#3 Training the model using predective analysis 

in this phase we import the appropriate libraries used in machine learning, linear regression and more 
then split the data for training then in tree sets 

## 3.1 This code uses the when function to check if the “Recommended ICU/Critical Care beds for Isolation” column contains only digits. 

If it does, it casts the column to an IntegerType. Otherwise, it sets the value to None

```python
from pyspark.sql.functions import when, col

df = df.withColumn("Recommended ICU/Critical Care beds for Isolation",
                   when(col("Recommended ICU/Critical Care beds for Isolation").rlike("^[0-9]+$"),
                        col("Recommended ICU/Critical Care beds for Isolation").cast(IntegerType())
                       ).otherwise(None))

```

### 3.1.1 Spliting the data

```python
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
```

### 3.1.2 Assembling vector features for linear regression purposes

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["Regular Isolation Beds Available", "Recommended ICU/Critical Care beds for Isolation"],
    outputCol="features"
)

train_vect = assembler.transform(train_df)
test_vect = assembler.transform(test_df)

```

## 4 Training the model 
```python
from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol=label_col)
fitted_model = lr.fit(train_vect)

```

### 4.1 Visualize the model 

here we import neccessary libraries like matplotlib which will help us visualize the linear regression of the model Trained 

```python
test_pred = fitted_model.transform(test_vect)
true_vals = test_pred.select("Recommended ICU Beds").collect()
pred_vals = test_pred.select("prediction").collect()

import matplotlib.pyplot as plt
true_x = [r.Recommended_ICU_Beds for r in true_vals]  
pred_x = [r.prediction for r in pred_vals]
plt.scatter(true_x, pred_x)
plt.plot([0, 50], [0, 50], c='red') 
plt.title("True vs Predicted")
plt.show()

```

![alt text](https://github.com/ImMwiti/covid-19-analysis-hadoop-pyspark/blob/main/Screenshots/model.png)

 ## 5 Testing the model 

```python
# Load test data
test_df = spark.read.format("csv").option("header","true").load("/path/to/test_data.csv")

# Import model 
from model import RandomForestModel
model = RandomForestModel() 

# Make predictions
predictions = model.transform(test_df)

# Evaluate predictions 
from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator()
rmse = evaluator.evaluate(predictions)

# Show first 5 predictions 
predictions.show(5)

# Print metrics
print("Root Mean Squared Error:", rmse)
```

