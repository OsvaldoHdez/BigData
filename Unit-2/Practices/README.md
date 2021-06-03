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

# Practice 2 - Decision tree classifier
Decision Tree algorithm belongs to the family of supervised learning algorithms. Unlike other supervised learning algorithms, the decision tree algorithm can be used for solving regression and classification problems too.

The goal of using a Decision Tree is to create a training model that can use to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).

```scala
// Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassificationModel
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Automatically identify categorical features, and index them.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)
// features with > 4 distinct values are treated as continuous.

 // Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a DecisionTree model.
val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labelsArray(0))

// Chain indexers and tree in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
```
# Practice 3 Random forest classifier
Random forest is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result.

```scala
// Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("RandomForestClassifierExample").getOrCreate()

// Load and parse the data file, converting it to a DataFrame.
 val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)

// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous.
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a RandomForest model.
val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labelsArray(0))

// Chain indexers and forest in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

// Train model. This also runs the indexers.
 val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
println(s"Learned classification forest model:\n ${rfModel.toDebugString}")


```

# Practice 4 Gradient boosted tree classifier.
Gradient boosting classifiers are a group of machine learning algorithms that combine many weak learning models together to create a strong predictive model. Decision trees are usually used when doing gradient boosting. Gradient boosting models are becoming popular because of their effectiveness at classifying complex datasets, and have recently been used to win many Kaggle data science competitions.
```scala
// Import libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

// Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("GradientBoostedTreeClassifierExample").getOrCreate()

// Load and parse the data file, converting it to a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(data)
    
// Automatically identify categorical features, and index them.
// Set maxCategories so features with > 4 distinct values are treated as continuous. 
val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(data)

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

// Train a GBT model.
val gbt = new GBTClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setMaxIter(10).setFeatureSubsetStrategy("auto")

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labelsArray(0))

// Chain indexers and GBT in a Pipeline.
val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, gbt, labelConverter))

// Train model. This also runs the indexers.
val model = pipeline.fit(trainingData)

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "label", "features").show(5)

// Select (prediction, true label) and compute test error.
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")
    
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1.0 - accuracy}")

val gbtModel = model.stages(2).asInstanceOf[GBTClassificationModel]
println(s"Learned classification GBT model:\n ${gbtModel.toDebugString}")
``` 
# Practice 5 Multilayer perceptron classifier
A multilayer perceptron (MLP) is a class of feedforward artificial neural network (ANN). 

An MLP consists of at least three layers of nodes: an input layer, a hidden layer and an output layer. Except for the input nodes, each node is a neuron that uses a nonlinear activation function. MLP utilizes a supervised learning technique called backpropagation for training.

```scala

// Import libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

// Split the data into train and test
val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
val train = splits(0)
val test = splits(1)

// specify layers for the neural network:
// input layer of size 4 (features), two intermediate of size 5 and 4
// and output of size 3 (classes)
val layers = Array[Int](4, 5, 4, 3)

// create the trainer and set its parameters
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// train the model
val model = trainer.fit(train)

// compute accuracy on the test set
val result = model.transform(test)
val predictionAndLabels = result.select("prediction", "label")
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")


```
# Practice 6 Linear Support Vector Machine
Support Vector Machine is a linear model for classification and regression problems. It can solve linear and non-linear problems and work well for many practical problems. The idea of SVM is simple: The algorithm creates a line or a hyperplane which separates the data into classes.


```scala
// Import the "LinearSVC" library, this binary classifier optimizes the hinge loss using the OWLQN optimizer. 
import org.apache.spark.ml.classification.LinearSVC

// Import session.
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("LinearSVCExample").getOrCreate()

// Load the training data. 
val training = spark.read.format("libsvm").load("/Archivos/sample_libsvm_data.txt")

// Set the maximum number of iterations and the regularization parameter .
val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

// Make a fit to adjust the model.
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercepts for the Linear SVC.
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

```
# Practice 7 One-vs-Rest classifier (a.k.a. One-vs-All)
One-vs-rest (OvR for short, also referred to as One-vs-All or OvA) is a heuristic method for using binary classification algorithms for multi-class classification.

It involves splitting the multi-class dataset into multiple binary classification problems. A binary classifier is then trained on each binary classification problem and predictions are made using the model that is the most confident.

```scala

// Import libraries
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName(s"OneVsRestExample").getOrCreate()

// load data file.
val inputData = spark.read.format("libsvm").load("data/mllib/sample_multiclass_classification_data.txt")

// generate the train/test split.
val Array(train, test) = inputData.randomSplit(Array(0.8, 0.2))

// instantiate the base classifier
 val classifier = new LogisticRegression().setMaxIter(10).setTol(1E-6).setFitIntercept(true)

// instantiate the One Vs Rest Classifier.
val ovr = new OneVsRest().setClassifier(classifier)

// train the multiclass model.
val ovrModel = ovr.fit(train)

// score the model on test data.
val predictions = ovrModel.transform(test)

// obtain evaluator.
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

// compute the classification error on test data.
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${1 - accuracy}")
```
# Practice 8  Naive Bayes
It is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

``` scala
// Import libraries
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("NaiveBayesExample").getOrCreate()

// Load the data stored in LIBSVM format as a DataFrame.
val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

// Split the data into training and test sets (30% held out for testing)
val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

// Train a NaiveBayes model.
val model = new NaiveBayes().fit(trainingData)

// Select example rows to display.
val predictions = model.transform(testData)
predictions.show()

// Select (prediction, true label) and compute test error
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test set accuracy = $accuracy")
```

