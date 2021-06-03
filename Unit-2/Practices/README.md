# Practices
---
## Practice 1 - Basic Statistics
### Correlation
Calculating the correlation between two series of data is a common operation in Statistics. In spark.ml we provide the flexibility to calculate pairwise correlations among many series. The supported correlation methods are currently Pearson’s and Spearman’s correlation.
#### Code
```scala
// 1. Importing the library for matrices and vectors 
import org.apache.spark.ml.linalg.{Matrix, Vectors}

// 2. Import correlation library 
import org.apache.spark.ml.stat.Correlation

// 3. Allows access to a single row value through generic ordinal access, as well as primitive access
import org.apache.spark.sql.Row

// 4. Session import and creation 
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("CorrelationExample").getOrCreate()

// 5. Importing the Implicit Conversion to Convert RDDs to DataFrames 
import spark.implicits._

// 6. Creation of dense and sparse vectors, within a matrix
val data = Seq(
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
)

// 7. The data is extracted from the matrix and a dataframe is created 
val df = data.map(Tuple1.apply).toDF("features")

// 8. Creation of the Pearson correlation matrix using the created dataframe 
val Row(coeff1: Matrix) = Correlation.corr(df, "features").head

// 9. Print result 
println(s"Pearson correlation matrix:\n $coeff1")

// 10. Creating the Spearman correlation matrix using the created dataframe 
val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head

// 11. Print result 
println(s"Spearman correlation matrix:\n $coeff2")
```
#### Results
```scala
// 1. Importing the library for matrices and vectors 
scala> import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.linalg.{Matrix, Vectors}

// 2. Import correlation library 
scala> import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.stat.Correlation

// 3. Allows access to a single row value through generic ordinal access, as well as primitive access
scala> import org.apache.spark.sql.Row
import org.apache.spark.sql.Row

// 4. Session import and creation 
scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
scala> val spark = SparkSession.builder.appName("CorrelationExample").getOrCreate()
21/04/29 14:40:51 WARN SparkSession$Builder: Using an existing SparkSession; some spark core configurations may not take effect.
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@77a14911

// 5. Importing the Implicit Conversion to Convert RDDs to DataFrames 
scala> import spark.implicits._
import spark.implicits._

// 6. Creation of dense and sparse vectors, within a matrix
scala> val data = Seq(
     |     Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
     |     Vectors.dense(4.0, 5.0, 0.0, 3.0),
     |     Vectors.dense(6.0, 7.0, 0.0, 8.0),
     |     Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
     | )
data: Seq[org.apache.spark.ml.linalg.Vector] = List((4,[0,3],[1.0,-2.0]), [4.0,5.0,0.0,3.0], [6.0,7.0,0.0,8.0], (4,[0,3],[9.0,1.0]))

// 7. The data is extracted from the matrix and a dataframe is created 
scala> val df = data.map(Tuple1.apply).toDF("features")
df: org.apache.spark.sql.DataFrame = [features: vector]

// 8. Creation of the Pearson correlation matrix using the created dataframe 
scala> val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
21/04/29 14:42:03 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/04/29 14:42:03 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
21/04/29 14:42:04 WARN PearsonCorrelation: Pearson correlation matrix contains NaN values.
coeff1: org.apache.spark.ml.linalg.Matrix =
1.0                   0.055641488407465814  NaN  0.4004714203168137
0.055641488407465814  1.0                   NaN  0.9135958615342522
NaN                   NaN                   1.0  NaN
0.4004714203168137    0.9135958615342522    NaN  1.0

// 9. Print result 
scala> println(s"Pearson correlation matrix:\n $coeff1")
Pearson correlation matrix:
 1.0                   0.055641488407465814  NaN  0.4004714203168137  
0.055641488407465814  1.0                   NaN  0.9135958615342522  
NaN                   NaN                   1.0  NaN                 
0.4004714203168137    0.9135958615342522    NaN  1.0   

// 10. Creating the Spearman correlation matrix using the created dataframe 
scala> val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
21/04/29 14:43:34 WARN PearsonCorrelation: Pearson correlation matrix contains NaN values.
coeff2: org.apache.spark.ml.linalg.Matrix =
1.0                  0.10540925533894532  NaN  0.40000000000000174
0.10540925533894532  1.0                  NaN  0.9486832980505141
NaN                  NaN                  1.0  NaN
0.40000000000000174  0.9486832980505141   NaN  1.0

// 11. Print result 
scala> println(s"Spearman correlation matrix:\n $coeff2")
Spearman correlation matrix:
 1.0                  0.10540925533894532  NaN  0.40000000000000174  
0.10540925533894532  1.0                  NaN  0.9486832980505141   
NaN                  NaN                  1.0  NaN                  
0.40000000000000174  0.9486832980505141   NaN  1.0 
```
### Hypothesis testing
Hypothesis testing is a powerful tool in statistics to determine whether a result is statistically significant, whether this result occurred by chance or not. spark.ml currently supports Pearson’s Chi-squared (χ2) tests for independence.

ChiSquareTest conducts Pearson’s independence test for every feature against the label. For each feature, the (feature, label) pairs are converted into a contingency matrix for which the Chi-squared statistic is computed. All label and feature values must be categorical.
#### Code
```scala
// 1. Importing the library for vectors
import org.apache.spark.ml.linalg.{Vector, Vectors}

// 2. ChiSquare library import 
import org.apache.spark.ml.stat.ChiSquareTest

// 3. Session import and creation 
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("ChiSquareTestExample").getOrCreate()

// 4. Importing the Implicit Conversion to Convert RDDs to DataFrames 
import spark.implicits._

// 5. Create dense vectors 
val data = Seq(
    (0.0, Vectors.dense(0.5, 10.0)),
    (0.0, Vectors.dense(1.5, 20.0)),
    (1.0, Vectors.dense(1.5, 30.0)),
    (0.0, Vectors.dense(3.5, 30.0)),
    (0.0, Vectors.dense(3.5, 40.0)),
    (1.0, Vectors.dense(3.5, 40.0))
)

// 6. Creation of the dataframe with the previous data 
val df = data.toDF("label", "features")

// 7. The first values of the dataframe are taken 
val chi = ChiSquareTest.test(df, "features", "label").head
```
#### Results
```scala
// 1. Importing the library for vectors
scala> import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.{Vector, Vectors}

// 2. ChiSquare library import 
scala> import org.apache.spark.ml.stat.ChiSquareTest
import org.apache.spark.ml.stat.ChiSquareTest

// 3. Session import and creation 
scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
scala> val spark = SparkSession.builder.appName("ChiSquareTestExample").getOrCreate()
21/04/29 15:00:00 WARN SparkSession$Builder: Using an existing SparkSession; some spark core configurations may not take effect.
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@7e2c8622

// 4. Importing the Implicit Conversion to Convert RDDs to DataFrames 
scala> import spark.implicits._
import spark.implicits._

// 5. Create dense vectors 
scala> val data = Seq(
     |     (0.0, Vectors.dense(0.5, 10.0)),
     |     (0.0, Vectors.dense(1.5, 20.0)),
     |     (1.0, Vectors.dense(1.5, 30.0)),
     |     (0.0, Vectors.dense(3.5, 30.0)),
     |     (0.0, Vectors.dense(3.5, 40.0)),
     |     (1.0, Vectors.dense(3.5, 40.0))
     | )
data: Seq[(Double, org.apache.spark.ml.linalg.Vector)] = List((0.0,[0.5,10.0]), (0.0,[1.5,20.0]), (1.0,[1.5,30.0]), (0.0,[3.5,30.0]), (0.0,[3.5,40.0]), (1.0,[3.5,40.0]))

// 6. Creation of the dataframe with the previous data 
scala> val df = data.toDF("label", "features")
df: org.apache.spark.sql.DataFrame = [label: double, features: vector]

// 7. The first values of the dataframe are taken 
scala> val chi = ChiSquareTest.test(df, "features", "label").head
chi: org.apache.spark.sql.Row = [[0.6872892787909721,0.6822703303362126],WrappedArray(2, 3),[0.75,1.5]]

// 8. We take the values of the dataframe
println(s"pValues = ${chi.getAs[Vector](0)}")
pValues = [0.6872892787909721,0.6822703303362126]

// 9. Next we will look for the grade of fredom of the model
println(s"degreesOfFreedom ${chi.getSeq[Int](1).mkString("[", ",", "]")}")
degreesOfFreedom [2,3]

// 10. Finally certain values are extracted from a given vector all based on the chi square function
println(s"statistics ${chi.getAs[Vector](2)}")
statistics [0.75,1.5]
```

### Summarizer
We provide vector column summary statistics for Dataframe through Summarizer. The available metrics are the maximum, minimum, mean, variance, and the number of non-zeros in columns, as well as the total count.

The Summarizer method is a good tool for getting various statistics on a new vector column when using machine learning pipelines. To use the Summarizer, import the package from pyspark.ml.stat import Summarizer into PySpark and import org.apache.spark.ml.stat Summarizer into Spark Scala.

#### Code
```scala
// 1. Importing the vector library 
import org.apache.spark.ml.linalg.{Vector, Vectors}

// 2. Import summarizer 
import org.apache.spark.ml.stat.Summarizer

// 3. Session import and creation 
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("SummarizerExample").getOrCreate()

// 4. Import of necessary libraries, in this use of vectors and the summarizer itself
import spark.implicits._    
import Summarizer._

// 5. Create a set of vectors or sequence 
val data = Seq(
  (Vectors.dense(2.0, 3.0, 5.0), 1.0),
  (Vectors.dense(4.0, 6.0, 7.0), 2.0)
)

// 6. Creation of the dataframe from the vectors
val df = data.toDF("features", "weight")

// 7. Use the summarizer library to obtain the mean and variance of some data in the requested dataframe
val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()

// 8. The variables previously worked on are printed
println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")

// 9. The process is repeated with 2 new variables
val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features")).as[(Vector, Vector)].first()

// 10. Variable printing
println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")
```
#### Results
```scala
// 1. Importing the vector library 
scala> import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linalg.{Vector, Vectors}

// 2. Import summarizer 
scala> import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.stat.Summarizer

// 3. Session import and creation 
scala> import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.SparkSession
scala> val spark = SparkSession.builder.appName("SummarizerExample").getOrCreate()
21/04/29 18:20:37 WARN SparkSession$Builder: Using an existing SparkSession; some spark core configurations may not take effect.
spark: org.apache.spark.sql.SparkSession = org.apache.spark.sql.SparkSession@4b02dc4e

// 4. Import of necessary libraries, in this use of vectors and the summarizer itself
scala> import spark.implicits._    
import spark.implicits._

scala> import Summarizer._
import Summarizer._

// 5. Create a set of vectors or sequence 
scala> val data = Seq(
     |   (Vectors.dense(2.0, 3.0, 5.0), 1.0),
     |   (Vectors.dense(4.0, 6.0, 7.0), 2.0)
     | )
data: Seq[(org.apache.spark.ml.linalg.Vector, Double)] = List(([2.0,3.0,5.0],1.0), ([4.0,6.0,7.0],2.0))

// 6. Creation of the dataframe from the vectors
scala> val df = data.toDF("features", "weight")
df: org.apache.spark.sql.DataFrame = [features: vector, weight: double]

// 7. Use the summarizer library to obtain the mean and variance of some data in the requested dataframe
scala> val (meanVal, varianceVal) = df.select(metrics("mean", "variance").summary($"features", $"weight").as("summary")).select("summary.mean", "summary.variance").as[(Vector, Vector)].first()
meanVal: org.apache.spark.ml.linalg.Vector = [3.333333333333333,5.0,6.333333333333333]
varianceVal: org.apache.spark.ml.linalg.Vector = [2.0,4.5,2.0]

// 8. The variables previously worked on are printed
scala> println(s"with weight: mean = ${meanVal}, variance = ${varianceVal}")
with weight: mean = [3.333333333333333,5.0,6.333333333333333], variance = [2.0,4.5,2.0]

// 9. The process is repeated with 2 new variables
scala> val (meanVal2, varianceVal2) = df.select(mean($"features"), variance($"features")).as[(Vector, Vector)].first()
meanVal2: org.apache.spark.ml.linalg.Vector = [3.0,4.5,6.0]
varianceVal2: org.apache.spark.ml.linalg.Vector = [2.0,4.5,2.0]

// 10. Variable printing
scala> println(s"without weight: mean = ${meanVal2}, sum = ${varianceVal2}")
without weight: mean = [3.0,4.5,6.0], sum = [2.0,4.5,2.0]
```

## Practice 2 - Decision tree classifier
Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).
#### Code
```scala
// 1. Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

// 3. Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// 4. Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// 5. Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
// features with > 4 distinct values are treated as continuous.

// 6. Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// 7. Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// 8. Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// 9. Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// 10. Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// 11. Make predictions.
val predictions = model.transform(testData)

// 12. Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// 13. Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```
#### Results
```scala
// 4. Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
scala> val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_bf49ec2354c0

// 5. Automatically identify categorical features, and index them.
scala> val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_34d27447c8dc

// 6. Split the data into training and test sets (30% held out for testing).
scala> val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

// 7. Train a DecisionTree model.
scala> val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
dt: org.apache.spark.ml.classification.DecisionTreeClassifier = dtc_01f998a611a6

// 8. Convert indexed labels back to original labels.
scala> val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_40e00e913d15

// 9. Chain indexers and tree in a Pipeline.
scala> val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))
pipeline: org.apache.spark.ml.Pipeline = pipeline_35625353eb86

// 10. Train model. This also runs the indexers.
scala> val model = pipeline.fit(trainingData)
model: org.apache.spark.ml.PipelineModel = pipeline_35625353eb86

// 11. Make predictions.
scala> val predictions = model.transform(testData)
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]

// 12. Select example rows to display.
scala> predictions.select("predictedLabel", "label", "features").show(5)
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[121,122,123...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[126,127,128...|
|           0.0|  0.0|(692,[126,127,128...|
+--------------+-----+--------------------+
only showing top 5 rows

// 13. Select (prediction, true label) and compute test error.
scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_f31b0dfffda8

scala>     
     | val accuracy = evaluator.evaluate(predictions)
accuracy: Double = 1.0

scala> println(s"Test Error = ${(1.0 - accuracy)}")
Test Error = 0.0

scala> val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
treeModel: org.apache.spark.ml.classification.DecisionTreeClassificationModel = DecisionTreeClassificationModel (uid=dtc_01f998a611a6) of depth 2 with 5 nodes

scala> println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
Learned classification tree model:
 DecisionTreeClassificationModel (uid=dtc_01f998a611a6) of depth 2 with 5 nodes
  If (feature 434 <= 70.5)
   If (feature 99 in {2.0})
    Predict: 0.0
   Else (feature 99 not in {2.0})
    Predict: 1.0
  Else (feature 434 > 70.5)
   Predict: 0.0
```
## Practice 3 - Random forest classifier
Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.
#### Code
```scala
// 1. Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()

// 3. Load and parse the data file, converting it to a DataFrame.
 val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// 4. Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// 5. Automatically identify categorical features, and index them.
//    Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// 6. Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// 7. Train a RandomForest model.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// 8. Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// 9. Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// 10. Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// 11. Make predictions.
val predictions = model.transform(testData)

// 12. Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// 13. Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
```
#### Results
```scala
// 4. Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
scala> val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_e70a822086a9

// 5. Automatically identify categorical features, and index them.
//    Set maxCategories so features with > 4 distinct values are treated as continuous.
scala> val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_789b61145ba6

// 6. Split the data into training and test sets (30% held out for testing).
scala> val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

// 7. Train a RandomForest model.
scala> val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)
rf: org.apache.spark.ml.classification.RandomForestClassifier = rfc_b1143d9c1f7a

// 8. Convert indexed labels back to original labels.
scala> val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_2fb425014a5b

// 9. Chain indexers and forest in a Pipeline.
scala> val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))
pipeline: org.apache.spark.ml.Pipeline = pipeline_cf23008ddadf

// 10. Train model. This also runs the indexers.
scala> val model = pipeline.fit(trainingData)
model: org.apache.spark.ml.PipelineModel = pipeline_cf23008ddadf

// 11. Make predictions.
scala> val predictions = model.transform(testData)
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]

// 12. Select example rows to display.
scala> predictions.select("predictedLabel", "label", "features").show(5)
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[123,124,125...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[125,126,127...|
|           0.0|  0.0|(692,[126,127,128...|
+--------------+-----+--------------------+
only showing top 5 rows

// 13. Select (prediction, true label) and compute test error.
scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_76b3dab35979

scala> val accuracy = evaluator.evaluate(predictions)
accuracy: Double = 1.0

scala> println(s"Test Error = ${(1.0 - accuracy)}")
Test Error = 0.0

scala> val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
rfModel: org.apache.spark.ml.classification.RandomForestClassificationModel = RandomForestClassificationModel (uid=rfc_b1143d9c1f7a) with 10 trees

scala> println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
Learned classification forest model:
 RandomForestClassificationModel (uid=rfc_b1143d9c1f7a) with 10 trees
  Tree 0 (weight 1.0):
    If (feature 412 <= 8.0)
     If (feature 443 <= 6.5)
      Predict: 0.0
     Else (feature 443 > 6.5)
      Predict: 1.0
    Else (feature 412 > 8.0)
     Predict: 1.0
  Tree 1 (weight 1.0):
    If (feature 463 <= 2.0)
     If (feature 456 <= 31.5)
      If (feature 274 <= 6.0)
       Predict: 0.0
      Else (feature 274 > 6.0)
       Predict: 1.0
     Else (feature 456 > 31.5)
      Predict: 1.0
    Else (feature 463 > 2.0)
     Predict: 0.0
  Tree 2 (weight 1.0):
    If (feature 385 <= 4.0)
     If (feature 317 <= 8.0)
      If (feature 489 <= 1.5)
       If (feature 270 <= 6.5)
        Predict: 0.0
       Else (feature 270 > 6.5)
        Predict: 1.0
      Else (feature 489 > 1.5)
       Predict: 0.0
     Else (feature 317 > 8.0)
      If (feature 630 <= 5.0)
       Predict: 1.0
      Else (feature 630 > 5.0)
       Predict: 0.0
    Else (feature 385 > 4.0)
     Predict: 1.0
  Tree 3 (weight 1.0):
    If (feature 328 <= 25.5)
     If (feature 350 <= 7.0)
      If (feature 179 <= 3.0)
       Predict: 0.0
      Else (feature 179 > 3.0)
       Predict: 1.0
     Else (feature 350 > 7.0)
      Predict: 0.0
    Else (feature 328 > 25.5)
     Predict: 1.0
  Tree 4 (weight 1.0):
    If (feature 429 <= 23.5)
     If (feature 245 <= 16.0)
      Predict: 0.0
     Else (feature 245 > 16.0)
      Predict: 1.0
    Else (feature 429 > 23.5)
     Predict: 1.0
  Tree 5 (weight 1.0):
    If (feature 462 <= 62.5)
     Predict: 1.0
    Else (feature 462 > 62.5)
     Predict: 0.0
  Tree 6 (weight 1.0):
    If (feature 512 <= 1.5)
     If (feature 545 <= 3.0)
      If (feature 157 <= 11.5)
       Predict: 0.0
      Else (feature 157 > 11.5)
       Predict: 1.0
     Else (feature 545 > 3.0)
      If (feature 482 <= 5.5)
       Predict: 0.0
      Else (feature 482 > 5.5)
       Predict: 1.0
    Else (feature 512 > 1.5)
     Predict: 1.0
  Tree 7 (weight 1.0):
    If (feature 512 <= 1.5)
     If (feature 510 <= 2.5)
      Predict: 0.0
     Else (feature 510 > 2.5)
      Predict: 1.0
    Else (feature 512 > 1.5)
     Predict: 1.0
  Tree 8 (weight 1.0):
    If (feature 462 <= 62.5)
     If (feature 324 <= 251.5)
      Predict: 1.0
     Else (feature 324 > 251.5)
      Predict: 0.0
    Else (feature 462 > 62.5)
     Predict: 0.0
  Tree 9 (weight 1.0):
    If (feature 377 <= 34.0)
     If (feature 492 <= 140.5)
      Predict: 1.0
     Else (feature 492 > 140.5)
      Predict: 0.0
    Else (feature 377 > 34.0)
     If (feature 413 <= 11.0)
      Predict: 0.0
     Else (feature 413 > 11.0)
      Predict: 1.0
```
## Practice 4 - Gradient boosted tree classifier.
Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting. Gradient boosting models are becoming popular because of their effectiveness at classifying complex datasets, and have recently been used to win many Kaggle data science competitions.
#### Code
```scala
// 1. Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("GradientBoostedTreeClassifierExample").getOrCreate()

// 3. Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// 4. Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    
// 5. Automatically identify categorical features, and index them.
//    Set maxCategories so features with > 4 distinct values are treated as continuous. 
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// 6. Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// 7. Train a GBT model.
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

// 8. Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

// 9. Chain indexers and GBT in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// 10. Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// 11. Make predictions.
val predictions = model.transform(testData)

// 12. Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// 13. Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
``` 
#### Results
```scala
// 4. Index labels, adding metadata to the label column.
//    Fit on whole dataset to include all labels in index.
scala> val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
labelIndexer: org.apache.spark.ml.feature.StringIndexerModel = strIdx_753b860253d6
    
// 5. Automatically identify categorical features, and index them.
//    Set maxCategories so features with > 4 distinct values are treated as continuous. 
scala> val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
featureIndexer: org.apache.spark.ml.feature.VectorIndexerModel = vecIdx_e80a3f8f2068

// 6. Split the data into training and test sets (30% held out for testing).
scala> val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

// 7. Train a GBT model.
scala> val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")
gbt: org.apache.spark.ml.classification.GBTClassifier = gbtc_7425b01cf3c0

// 8. Convert indexed labels back to original labels.
scala> val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)
labelConverter: org.apache.spark.ml.feature.IndexToString = idxToStr_3372b464727e

// 9. Chain indexers and GBT in a Pipeline.
scala> val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))
pipeline: org.apache.spark.ml.Pipeline = pipeline_66198b9fce2c

// 10. Train model. This also runs the indexers.
scala> val model = pipeline.fit(trainingData)
model: org.apache.spark.ml.PipelineModel = pipeline_66198b9fce2c

// 11. Make predictions.
scala> val predictions = model.transform(testData)
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 6 more fields]

// 12. Select example rows to display.
scala> predictions.select("predictedLabel", "label", "features").show(5)
21/06/03 10:01:41 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/06/03 10:01:41 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
+--------------+-----+--------------------+
|predictedLabel|label|            features|
+--------------+-----+--------------------+
|           0.0|  0.0|(692,[95,96,97,12...|
|           0.0|  0.0|(692,[123,124,125...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[124,125,126...|
|           0.0|  0.0|(692,[126,127,128...|
+--------------+-----+--------------------+
only showing top 5 rows

// 13. Select (prediction, true label) and compute test error.
scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_65c038432427

scala> val accuracy = evaluator.evaluate(predictions)
accuracy: Double = 0.9230769230769231

scala> println(s"Test Error = ${1.0 - accuracy}")
Test Error = 0.07692307692307687

scala> val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
gbtModel: org.apache.spark.ml.classification.GBTClassificationModel = GBTClassificationModel (uid=gbtc_7425b01cf3c0) with 10 trees

scala> println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
Learned classification GBT model:
 GBTClassificationModel (uid=gbtc_7425b01cf3c0) with 10 trees
  Tree 0 (weight 1.0):
    If (feature 378 <= 90.5)
     If (feature 99 in {2.0})
      Predict: -1.0
     Else (feature 99 not in {2.0})
      Predict: 1.0
    Else (feature 378 > 90.5)
     Predict: -1.0
  Tree 1 (weight 0.1):
    If (feature 378 <= 90.5)
     If (feature 239 <= 253.5)
      If (feature 300 <= 254.5)
       Predict: 0.47681168808847024
      Else (feature 300 > 254.5)
       Predict: 0.4768116880884712
     Else (feature 239 > 253.5)
      Predict: -0.4768116880884712
    Else (feature 378 > 90.5)
     Predict: -0.47681168808847013
  Tree 2 (weight 0.1):
    If (feature 378 <= 90.5)
     If (feature 549 <= 253.5)
      Predict: 0.4381935810427206
     Else (feature 549 > 253.5)
      Predict: -0.43819358104271977
    Else (feature 378 > 90.5)
     If (feature 296 <= 114.5)
      Predict: -0.4381935810427206
     Else (feature 296 > 114.5)
      Predict: -0.43819358104272066
  Tree 3 (weight 0.1):
    If (feature 433 <= 52.5)
     If (feature 100 <= 193.5)
      Predict: 0.40514968028459825
     Else (feature 100 > 193.5)
      Predict: -0.4051496802845982
    Else (feature 433 > 52.5)
     Predict: -0.40514968028459825
  Tree 4 (weight 0.1):
    If (feature 490 <= 43.0)
     If (feature 100 <= 193.5)
      If (feature 150 <= 217.5)
       Predict: 0.37658413183529926
      Else (feature 150 > 217.5)
       Predict: 0.3765841318352994
     Else (feature 100 > 193.5)
      Predict: -0.3765841318352994
    Else (feature 490 > 43.0)
     Predict: -0.3765841318352992
  Tree 5 (weight 0.1):
    If (feature 433 <= 52.5)
     If (feature 99 in {2.0})
      Predict: -0.35166478958100994
     Else (feature 99 not in {2.0})
      If (feature 209 <= 206.0)
       Predict: 0.35166478958101005
      Else (feature 209 > 206.0)
       Predict: 0.3516647895810101
    Else (feature 433 > 52.5)
     Predict: -0.35166478958101
  Tree 6 (weight 0.1):
    If (feature 434 <= 70.5)
     If (feature 344 <= 253.5)
      If (feature 241 <= 21.5)
       Predict: 0.32974984655529926
      Else (feature 241 > 21.5)
       Predict: 0.3297498465552993
     Else (feature 344 > 253.5)
      Predict: -0.3297498465552984
    Else (feature 434 > 70.5)
     Predict: -0.3297498465552992
  Tree 7 (weight 0.1):
    If (feature 406 <= 100.5)
     If (feature 239 <= 253.5)
      If (feature 656 <= 32.0)
       If (feature 603 <= 247.5)
        Predict: 0.3103372455197956
       Else (feature 603 > 247.5)
        Predict: 0.3103372455197957
      Else (feature 656 > 32.0)
       Predict: 0.31033724551979563
     Else (feature 239 > 253.5)
      Predict: -0.31033724551979525
    Else (feature 406 > 100.5)
     If (feature 295 <= 126.0)
      If (feature 125 <= 7.5)
       Predict: -0.3103372455197956
      Else (feature 125 > 7.5)
       Predict: -0.3103372455197957
     Else (feature 295 > 126.0)
      Predict: -0.31033724551979563
  Tree 8 (weight 0.1):
    If (feature 378 <= 90.5)
     If (feature 239 <= 253.5)
      If (feature 214 <= 79.5)
       Predict: 0.2930291649125433
      Else (feature 214 > 79.5)
       If (feature 272 <= 115.5)
        Predict: 0.2930291649125433
       Else (feature 272 > 115.5)
        Predict: 0.2930291649125434
     Else (feature 239 > 253.5)
      Predict: -0.29302916491254294
    Else (feature 378 > 90.5)
     If (feature 462 <= 182.0)
      Predict: -0.2930291649125433
     Else (feature 462 > 182.0)
      Predict: -0.2930291649125434
  Tree 9 (weight 0.1):
    If (feature 462 <= 62.5)
     If (feature 239 <= 253.5)
      If (feature 182 <= 80.0)
       If (feature 370 <= 2.0)
        Predict: 0.27750666438358246
       Else (feature 370 > 2.0)
        Predict: 0.2775066643835825
      Else (feature 182 > 80.0)
       Predict: 0.27750666438358257
     Else (feature 239 > 253.5)
      Predict: -0.27750666438358174
    Else (feature 462 > 62.5)
     If (feature 379 <= 221.5)
      Predict: -0.2775066643835825
     Else (feature 379 > 221.5)
      Predict: -0.27750666438358257
``` 
## Practice 5 - Multilayer perceptron classifier
A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). 

An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.
#### Code
```scala
// 1. Import libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

// 3. Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

// 4. Split the data into train and test
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// 5. specify layers for the neural network:
//    input layer of size 4 (features), two intermediate of size 5 and 4
//    and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)

// 6. create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// 7. train the model
val model = trainer.fit(train)

// 8. compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
```
#### Results
```scala
// 4. Split the data into train and test
scala> val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
splits: Array[org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]] = Array([label: double, features: vector], [label: double, features: vector])

scala> val train = splits(0)
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

scala> val test = splits(1)
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

// 5. specify layers for the neural network:
//    input layer of size 4 (features), two intermediate of size 5 and 4
//    and output of size 3 (classes)
scala> val layers = Array[Int](4, 5, 4, 3)
layers: Array[Int] = Array(4, 5, 4, 3)

// 6. create the trainer and set its parameters
scala> val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)
trainer: org.apache.spark.ml.classification.MultilayerPerceptronClassifier = mlpc_6f162687f682

// 7. train the model
scala> val model = trainer.fit(train)
21/06/03 10:10:23 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/06/03 10:10:23 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
model: org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel = mlpc_6f162687f682

// 8. compute accuracy on the test set
scala> val result = model.transform(test)
result: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 3 more fields]

scala> val predictionAndLabels = result.select("prediction", "label")
predictionAndLabels: org.apache.spark.sql.DataFrame = [prediction: double, label: double]

scala> val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_1407b467dc12

scala> println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
Test set accuracy = 0.9019607843137255
```
## Practice 6 - Linear Support Vector Machine
Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.
#### Code
```scala
// 1. Import the "LinearSVC" library, this binary classifier optimizes the hinge loss using the OWLQN optimizer. 
import org.apache.spark.ml.classification.LinearSVC

// 2. Import session.
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

// 3. Load the training data. 
val training = spark.read.format("libsvm").load("/Archivos/sample_libsvm_data.txt")

// 4. Set the maximum number of iterations and the regularization parameter .
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// 5. Make a fit to adjust the model.
val lsvcModel = lsvc.fit(training)

// 6. Print the coefficients and intercepts for the Linear SVC.
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
```
#### Results
```scala
// 4. Set the maximum number of iterations and the regularization parameter .
scala> val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)
lsvc: org.apache.spark.ml.classification.LinearSVC = linearsvc_5cfeb310b35a

// 5. Make a fit to adjust the model.
scala> val lsvcModel = lsvc.fit(training)
21/06/03 10:14:21 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/06/03 10:14:21 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
lsvcModel: org.apache.spark.ml.classification.LinearSVCModel = linearsvc_5cfeb310b35a

// 6. Print the coefficients and intercepts for the Linear SVC.
scala> println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")
Coefficients: [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.170630317473439E-4,-1.172288654973735E-4,-8.882754836918948E-5,8.522360710187464E-5,0.0,0.0,-1.3436361263314267E-5,3.729569801338091E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,8.888949552633658E-4,2.9864059761812683E-4,3.793378816193159E-4,-1.762328898254081E-4,0.0,1.5028489269747836E-6,1.8056041144946687E-6,1.8028763260398597E-6,-3.3843713506473646E-6,-4.041580184807502E-6,2.0965017727015125E-6,8.536111642989494E-5,2.2064177429604464E-4,2.1677599940575452E-4,-5.472401396558763E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,9.21415502407147E-4,3.1351066886882195E-4,2.481984318412822E-4,0.0,-4.147738197636148E-5,-3.6832150384497175E-5,0.0,-3.9652366184583814E-6,-5.1569169804965594E-5,-6.624697287084958E-5,-2.182148650424713E-5,1.163442969067449E-5,-1.1535211416971104E-6,3.8138960488857075E-5,1.5823711634321492E-6,-4.784013432336632E-5,-9.386493224111833E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,4.3174897827077767E-4,1.7055492867397665E-4,0.0,-2.7978204136148868E-5,-5.88745220385208E-5,-4.1858794529775E-5,-3.740692964881002E-5,-3.9787939304887E-5,-5.545881895011037E-5,-4.505015598421474E-5,-3.214002494749943E-6,-1.6561868808274739E-6,-4.416063987619447E-6,-7.9986183315327E-6,-4.729962112535003E-5,-2.516595625914463E-5,-3.6407809279248066E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.4719098130614967E-4,0.0,-3.270637431382939E-5,-5.5703407875748054E-5,-5.2336892125702286E-5,-7.829604482365818E-5,-7.60385448387619E-5,-8.371051301348216E-5,-1.8669558753795108E-5,0.0,1.2045309486213725E-5,-2.3374084977016397E-5,-1.0788641688879534E-5,-5.5731194431606874E-5,-7.952979033591137E-5,-1.4529196775456057E-5,8.737948348132623E-6,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0012589360772978808,-1.816228630214369E-4,-1.0650711664557365E-4,-6.040355527710781E-5,-4.856392973921569E-5,-8.973895954652451E-5,-8.78131677062384E-5,-5.68487774673792E-5,-3.780926734276347E-5,1.3834897036553787E-5,7.585485129441565E-5,5.5017411816753975E-5,-1.5430755398169695E-5,-1.834928703625931E-5,-1.0354008265646844E-4,-1.3527847721351194E-4,-1.1245007647684532E-4,-2.9373916056750564E-5,-7.311217847336934E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.858228613863785E-4,-1.2998173971449976E-4,-1.478408021316135E-4,-8.203374605865772E-5,-6.556685320008032E-5,-5.6392660386580244E-5,-6.995571627330911E-5,-4.664348159856693E-5,-2.3026593698824318E-5,7.398833979172035E-5,1.4817176130099997E-4,1.0938317435545486E-4,7.940425167011364E-5,-6.743294804348106E-7,-1.2623302721464762E-4,-1.9110387355357616E-4,-1.8611622108961136E-4,-1.2776766254736952E-4,-8.935302806524433E-5,-1.239417230441996E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-2.829530831354112E-4,-1.3912189600461263E-4,-1.2593136464577562E-4,-5.964745187930992E-5,-5.360328152341982E-5,-1.0517880662090183E-4,-1.3856124131005022E-4,-7.181032974125911E-5,2.3249038865093483E-6,1.566964269571967E-4,2.3261206954040812E-4,1.7261638232256968E-4,1.3857530960270466E-4,-1.396299028868332E-5,-1.5765773982418597E-4,-2.0728798812007546E-4,-1.9106441272002828E-4,-1.2744834161431415E-4,-1.2755611630280015E-4,-5.1885591560478935E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.59081567023441E-4,-1.216531230287931E-4,-5.623851079809818E-5,-3.877987126382982E-5,-7.550900509956966E-5,-1.0703140005463545E-4,-1.4720428138106226E-4,-8.781423374509368E-5,7.941655609421792E-5,2.3206354986219992E-4,2.7506982343672394E-4,2.546722233188043E-4,1.810821666388498E-4,-1.3069916689929984E-5,-1.842374220886751E-4,-1.977540482445517E-4,-1.7722074063670741E-4,-1.487987014723575E-4,-1.1879021431288621E-4,-9.755283887790393E-5,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.302740311359312E-4,-5.3683030235535024E-5,-1.7631200013656873E-5,-7.846611034608254E-5,-1.22100767283256E-4,-1.7281968533449702E-4,-1.5592346128894157E-4,-5.239579492910452E-5,1.680719343542442E-4,2.8930086786548053E-4,3.629921493231646E-4,2.958223512266975E-4,2.1770466955449064E-4,-6.40884808188951E-5,-1.9058225556007997E-4,-2.0425138564600712E-4,-1.711994903702119E-4,-1.3853486798341369E-4,-1.3018592950855062E-4,-1.1887779512760102E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-7.021411112285498E-5,-1.694500843168125E-5,-7.189722824172193E-5,-1.4560828004346436E-4,-1.4935497340563198E-4,-1.9496419340776972E-4,-1.7383743417254187E-4,-3.3438825792010694E-5,2.866538327947017E-4,2.9812321570739803E-4,3.77250607691119E-4,3.211702827486386E-4,2.577995115175486E-4,-1.6627385656703205E-4,-1.8037105851523224E-4,-2.0419356344211325E-4,-1.7962237203420184E-4,-1.3726488083579862E-4,-1.3461014473741762E-4,-1.2264216469164138E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0015239752514658556,-5.472330865993813E-5,-9.65684394936216E-5,-1.3424729853486994E-4,-1.4727467799568E-4,-1.616270978824712E-4,-1.8458259010029364E-4,-1.9699647135089726E-4,1.3085261294290817E-4,2.943178857107149E-4,3.097773692834126E-4,4.112834769312103E-4,3.4113620757035025E-4,1.6529945924367265E-4,-2.1065410862650534E-4,-1.883924081539624E-4,-1.979586414569358E-4,-1.762131187223702E-4,-1.272343622678854E-4,-1.2708161719220297E-4,-1.4812221011889967E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.001140680600536578,-1.323467421269896E-4,-1.2904607854274846E-4,-1.4104748544921958E-4,-1.5194605434027872E-4,-2.1104539389774283E-4,-1.7911827582001795E-4,-1.8952948277194435E-4,2.1767571552539842E-4,3.0201791656326465E-4,4.002863274397723E-4,4.0322806756364006E-4,4.118077382608461E-4,3.7917405252859545E-6,-1.9886290660234838E-4,-1.9547443112937263E-4,-1.9857348218680872E-4,-1.3336892200703206E-4,-1.2830129292910815E-4,-1.1855916317355505E-4,-1.765597203760205E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-0.0010938769592297973,-1.2785475305234688E-4,-1.3424699777466666E-4,-1.505200652479287E-4,-1.9333287822872713E-4,-2.0385160086594937E-4,-1.7422470698847553E-4,4.63598443910652E-5,2.0617623087127652E-4,2.862882891134514E-4,4.074830988361515E-4,3.726357785147985E-4,3.507520190729629E-4,-1.516485494364312E-4,-1.7053751921469217E-4,-1.9638964654350848E-4,-1.9962586265806435E-4,-1.3612312664311173E-4,-1.218285533892454E-4,-1.1166712081624676E-4,-1.377283888177579E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.044386260118809E-4,-1.240836643202059E-4,-1.335317492716633E-4,-1.5783442604618277E-4,-1.9168434243384107E-4,-1.8710322733892716E-4,-1.1283989231463139E-4,1.1136504453105364E-4,1.8707244892705632E-4,2.8654279528966305E-4,4.0032117544983536E-4,3.169637536305377E-4,2.0158994278679014E-4,-1.3139392844616033E-4,-1.5181070482383948E-4,-1.825431845981843E-4,-1.602539928567571E-4,-1.3230404795396355E-4,-1.1669138691257469E-4,-1.0532154964150405E-4,-1.3709037042366007E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-4.0287410145021705E-4,-1.3563987950912995E-4,-1.3225887084018914E-4,-1.6523502389794188E-4,-2.0175074284706945E-4,-1.572459106394481E-4,2.577536501278673E-6,1.312463663419457E-4,2.0707422291927531E-4,3.9081065544314936E-4,3.3487058329898135E-4,2.5790441367156086E-4,2.6881819648016494E-5,-1.511383586714907E-4,-1.605428139328567E-4,-1.7267287462873575E-4,-1.1938943768052963E-4,-1.0505245038633314E-4,-1.1109385509034013E-4,-1.3469914274864725E-4,-2.0735223736035555E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.034374233912422E-4,-1.5961213688405883E-4,-1.274222123810994E-4,-1.582821104884909E-4,-2.1301220616286252E-4,-1.2933366375029613E-4,1.6802673102179614E-5,1.1020918082727098E-4,2.1160795272688753E-4,3.4873421050827716E-4,2.6487211944380384E-4,1.151606835026639E-4,-5.4682731396851946E-5,-1.3632001630934325E-4,-1.4340405857651405E-4,-1.248695773821634E-4,-8.462873247977974E-5,-9.580708414770257E-5,-1.0749166605399431E-4,-1.4618038459197777E-4,-3.7556446296204636E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-5.124342611878493E-4,-2.0369734099093433E-4,-1.3626985098328694E-4,-1.3313768183302705E-4,-1.871555537819396E-4,-1.188817315789655E-4,-1.8774817595622694E-5,5.7108412194993384E-5,1.2728161056121406E-4,1.9021458214915667E-4,1.2177397895874969E-4,-1.2461153574281128E-5,-7.553961810487739E-5,-1.0242174559410404E-4,-4.44873554195981E-5,-9.058561577961895E-5,-6.837347198855518E-5,-8.084409304255458E-5,-1.3316868299585082E-4,-2.0335916397646626E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.966510928472775E-4,-1.3738983629066386E-4,-3.7971221409699866E-5,-6.431763035574533E-5,-1.1857739882295322E-4,-9.359520863114822E-5,-5.0878371516215046E-5,-8.269367595092908E-8,0.0,1.3434539131099211E-5,-1.9601690213728576E-6,-2.8527045990494954E-5,-7.410332699310603E-5,-7.132130570080122E-5,-4.9780961185536E-5,-6.641505361384578E-5,-6.962005514093816E-5,-7.752898158331023E-5,-1.7393609499225025E-4,-0.0012529479255443958,0.0,0.0,2.0682521269893754E-4,0.0,0.0,0.0,0.0,0.0,-4.6702467383631055E-4,-1.0318036388792008E-4,1.2004408785841247E-5,0.0,-2.5158639357650687E-5,-1.2095240910793449E-5,-5.19052816902203E-6,-4.916790639558058E-6,-8.48395853563783E-6,-9.362757097074547E-6,-2.0959335712838412E-5,-4.7790091043859085E-5,-7.92797600958695E-5,-4.462687041778011E-5,-4.182992428577707E-5,-3.7547996285851254E-5,-4.52754480225615E-5,-1.8553562561513456E-5,-2.4763037962085644E-4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-3.4886180455242474E-4,-5.687523659359091E-6,7.380040279654313E-5,4.395860636703821E-5,7.145198242379862E-5,6.181248343370637E-6,0.0,-6.0855538083486296E-5,-4.8563908323274725E-5,-4.117920588930435E-5,-4.359283623112936E-5,-6.608754161500044E-5,-5.443032251266018E-5,-2.7782637880987207E-5,0.0,0.0,2.879461393464088E-4,-0.0028955529777851255,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.2312114837837392E-4,-1.9526747917254753E-5,-1.6999506829961688E-5,5.4835294148085086E-5,1.523441632762399E-5,-5.8365604525328614E-5,-1.2378194216521848E-4,-1.1750704953254656E-4,-6.19711523061306E-5,-5.042009645812091E-5,-1.4055260223565886E-4,-1.410330942465528E-4,-1.9272308238929396E-4,-4.802489964676616E-4] Intercept: 0.012911305214513969
```
## Practice 7 - One-vs-Rest classifier (a.k.a. One-vs-All)
One-vs-rest (OvR for short, also referred to as One-vs-All or OvA) is a heuristic method for using binary classification algorithms for multi-class classification.

It involves splitting the multi-class dataset into multiple binary classification problems. A binary classifier is then trained on each binary classification problem and predictions are made using the model that is the most confident.
#### Code
```scala
// 1. Import libraries
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName(s"OneVsRestExample").getOrCreate()

// 3. load data file.
val inputData = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

// 4. generate the train/test split.
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// 5. instantiate the base classifier
val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

// 6. instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(classifier)

// 7. train the multiclass model.
val ovrModel = ovr.fit(train)

// 8. score the model on test data.
val predictions = ovrModel.transform(test)

// 9. obtain evaluator.
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// 10. compute the classification error on test data.
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")
```
#### Results
```scala
// 4. generate the train/test split.
scala> val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))
train: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
test: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

// 5. instantiate the base classifier
scala> val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)
classifier: org.apache.spark.ml.classification.LogisticRegression = logreg_3e04c780357d

// 6. instantiate the One Vs Rest Classifier.
scala> val ovr = new OneVsRest().setClassifier(classifier)
ovr: org.apache.spark.ml.classification.OneVsRest = oneVsRest_624ac4afc455

// 7. train the multiclass model.
scala> val ovrModel = ovr.fit(train)
21/06/03 10:17:48 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/06/03 10:17:48 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
ovrModel: org.apache.spark.ml.classification.OneVsRestModel = oneVsRest_624ac4afc455

// 8. score the model on test data.
scala> val predictions = ovrModel.transform(test)
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 2 more fields]

// 9. obtain evaluator.
scala> val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_a44f2337859e

// 10. compute the classification error on test data.
scala> val accuracy = evaluator.evaluate(predictions)
accuracy: Double = 0.9642857142857143

scala> println(s"Test Error = ${1 - accuracy}")
Test Error = 0.0357142857142857
```
## Practice 8 - Naive Bayes
It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
#### Code
``` scala
// 1. Import libraries
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// 2. Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("NaiveBayesExample").getOrCreate()

// 3. Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("/home/valdo/Documentos/Gitkraken/BigData/Unit-2/Practices/Files/sample_libsvm_data.txt")

// 4. Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

// 5. Train a NaiveBayes model.
val model = new NaiveBayes().fit(trainingData)

// 6. Select example rows to display.
val predictions = model.transform(testData)
predictions.show()

// 7.Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")
```
#### Results
``` scala
// 4. Split the data into training and test sets (30% held out for testing)
scala> val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
trainingData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]
testData: org.apache.spark.sql.Dataset[org.apache.spark.sql.Row] = [label: double, features: vector]

// 5. Train a NaiveBayes model.
scala> val model = new NaiveBayes().fit(trainingData)
model: org.apache.spark.ml.classification.NaiveBayesModel = NaiveBayesModel (uid=nb_2cf012815544) with 2 classes

// 6. Select example rows to display.
scala> val predictions = model.transform(testData)
predictions: org.apache.spark.sql.DataFrame = [label: double, features: vector ... 3 more fields]

scala> predictions.show()
21/06/03 10:21:45 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
21/06/03 10:21:45 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
+-----+--------------------+--------------------+-----------+----------+
|label|            features|       rawPrediction|probability|prediction|
+-----+--------------------+--------------------+-----------+----------+
|  0.0|(692,[95,96,97,12...|[-173678.60946628...|  [1.0,0.0]|       0.0|
|  0.0|(692,[98,99,100,1...|[-178107.24302988...|  [1.0,0.0]|       0.0|
|  0.0|(692,[100,101,102...|[-100020.80519087...|  [1.0,0.0]|       0.0|
|  0.0|(692,[124,125,126...|[-183521.85526462...|  [1.0,0.0]|       0.0|
|  0.0|(692,[127,128,129...|[-183004.12461660...|  [1.0,0.0]|       0.0|
|  0.0|(692,[128,129,130...|[-246722.96394714...|  [1.0,0.0]|       0.0|
|  0.0|(692,[152,153,154...|[-208696.01108598...|  [1.0,0.0]|       0.0|
|  0.0|(692,[153,154,155...|[-261509.59951302...|  [1.0,0.0]|       0.0|
|  0.0|(692,[154,155,156...|[-217654.71748256...|  [1.0,0.0]|       0.0|
|  0.0|(692,[181,182,183...|[-155287.07585335...|  [1.0,0.0]|       0.0|
|  1.0|(692,[99,100,101,...|[-145981.83877498...|  [0.0,1.0]|       1.0|
|  1.0|(692,[100,101,102...|[-147685.13694275...|  [0.0,1.0]|       1.0|
|  1.0|(692,[123,124,125...|[-139521.98499849...|  [0.0,1.0]|       1.0|
|  1.0|(692,[124,125,126...|[-129375.46702012...|  [0.0,1.0]|       1.0|
|  1.0|(692,[126,127,128...|[-145809.08230799...|  [0.0,1.0]|       1.0|
|  1.0|(692,[127,128,129...|[-132670.15737290...|  [0.0,1.0]|       1.0|
|  1.0|(692,[128,129,130...|[-100206.72054749...|  [0.0,1.0]|       1.0|
|  1.0|(692,[129,130,131...|[-129639.09694930...|  [0.0,1.0]|       1.0|
|  1.0|(692,[129,130,131...|[-143628.65574273...|  [0.0,1.0]|       1.0|
|  1.0|(692,[129,130,131...|[-129238.74023248...|  [0.0,1.0]|       1.0|
+-----+--------------------+--------------------+-----------+----------+
only showing top 20 rows

// 7.Select (prediction, true label) and compute test error
scala> val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
evaluator: org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator = mcEval_ddba7410870d

scala> val accuracy = evaluator.evaluate(predictions)
accuracy: Double = 1.0

scala> println(s"Test set accuracy = $accuracy")
Test set accuracy = 1.0
```
