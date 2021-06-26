# Unit-4 Final Project
---
## Contents
- ### [Introduction](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#introduction-1)
- ### [Theoretical framework](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#theoretical-framework-1)
    - #### [Support Vector Machine](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#support-vector-machine-1)
    - #### [Decision Tree](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#decision-tree-1)
    - #### [Logistic Regression](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#logistic-regression-1)
    - #### [Multilayer Perceptron](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#multilayer-perceptron-1)
- ### [Implementation](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#implementation-1)
- ### [Results](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#results-1)
- ### [Conclusions](https://github.com/OsvaldoHdez/BigData/tree/main/Unit-4%20FP#conclusions-1)
---
## Introduction
The objective of this document is to perform a performance comparison of different classification algorithms, such as Linear Support Vector Machine, Decision Tree, Logistic Regression and Multilayer Perceptron, through apache spark with scala, running the algorithms around thirty times each, in order to make a better performance comparison between each of the machine learning algorithms. 

## Theoretical framework
- ### Support Vector Machine
    Support Vector Machine is one of the classic machine learning techniques that can still help solve big data classification problems. Especially, it can help multi-domain applications in a big data environment. 
    La máquina de vectores de soporte (SVM) es un clasificador lineal binario no probabilístico desarrollado de acuerdo con la minimización del riesgo estructural y el aprendizaje estadístico. Las SVM utilizan un proceso de aprendizaje supervisado para generar funciones de mapeo de entrada-salida a partir de los datos de entrada.

    **Some features:**
    - SVM offers a principles-based approach to machine learning problems due to its mathematical foundation in statistical learning theory.
    - SVM builds its solution in terms of a subset of the training input.
    - SVM has been widely used for classification, regression, novelty detection, and feature reduction tasks.

    #### Functioning
    Una máquina de vectores de soporte construye un hiperplano o un conjunto de hiperplanos en un espacio de dimensión alta o infinita, que se puede utilizar para clasificación, regresión u otras tareas. Intuitivamente, se logra una buena separación por el hiperplano que tiene la mayor distancia a los puntos de datos de entrenamiento más cercanos de cualquier clase (el llamado margen funcional), ya que en general, cuanto mayor es el margen, menor es el error de generalización del clasificador.

- ### Decision Tree
    Decision trees and their sets are popular methods for machine learning regression and classification tasks. Decision trees are widely used because they are easy to interpret, handle categorical features, extend to multiclass classification settings, require no feature scaling, and can capture feature non-linearities and interactions. Tree set algorithms, such as random forests and momentum, are among the best for classification and regression tasks. 

    **Features of a decision tree**
    - The decision tree consists of nodes that form a rooted tree, which means that it is a directed tree with a node called a "root" that has no leading edges.
    - All other nodes have exactly one leading edge. A node with leading edges is called an internal or test node.
    - All other nodes are called leaves (also known as decision or terminal nodes).
    - In a decision tree, each internal node divides the instance space into two or more subspaces according to a certain discrete function of the values of the input attributes. 
    
    #### Functioning
    In the simplest and most common case, each test considers a single attribute, so the instance space is partitioned according to the value of the attribute. In the case of numeric attributes, the condition refers to a range.

    Each sheet is assigned to a class that represents the most appropriate target value. Alternatively, the sheet can contain a probability vector indicating the probability that the target attribute has a certain value. Instances are classified by navigating from the root of the tree to a leaf, according to the results of the tests along the path. 

    <html><div align="center"><img src="https://i.ibb.co/hMRxqzR/Screenshot-2021-06-23-at-10-07-42-Proyecto-final.png"></div></html>

- ### Logistic Regression
    Logistic regression is a statistical instrument for multivariate analysis, of both explanatory and predictive use. Its use is useful when there is a dichotomous dependent variable (an attribute whose absence or presence we have scored with the values zero and one, respectively) and a set of predictive or independent variables, which can be quantitative (which are called covariates or covariates). or categorical. In the latter case, it is required that they be transformed into "dummy" variables, that is, simulated variables. 

    #### Purpose
    The purpose of the analysis is to: predict the probability that a certain “event” will happen to someone: for example, being unemployed = 1 or not being unemployed = 0, being poor = 1 or not poor = 0, receiving a sociologist = 1 or not received = 0).

    Determine which variables weigh more to increase or decrease the probability that the event in question will happen to someone. 

    #### Example
    For example, the logistic regression will take into account the values assumed in a series of variables (age, sex, educational level, position in the home, migratory origin, etc.) the subjects who are effectively unemployed (= 1) and those who they are not (= 0). Based on this, it will predict to each of the subjects - regardless of their real and current state - a certain probability of being unemployed (that is, of having a value of 1 in the dependent variable). 

- ### Multilayer Perceptron
    The multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network.

    The multilayer perceptron is the best known and most widely used type of neural network. In most cases, signals are transmitted within the network in one direction: from input to output. There is no loop, the output of each neuron does not affect the neuron itself. This architecture is called feed-forward and can be seen in the following image.

    <html><div align="center"><img src="https://i.ibb.co/FXvqSZB/Screenshot-2021-06-23-at-10-12-59-Proyecto-final.png"></div></html>

    The power of the multilayer perceptron comes precisely from non-linear activation functions. Almost any nonlinear function can be used for this purpose, except polynomial functions. Currently, the most widely used functions today are the unipolar (or logistic) sigmoid.

    #### Functioning
    MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. The nodes in the input layer represent the input data.
    All other nodes map inputs to outputs by linearly combining the inputs with the node's weights and bias and applying a trigger function.

    #### Some applications
    The multilayer perceptron is an example of an artificial neural network that is widely used for solving a number of different problems, including pattern recognition and interpolation.  

## Implementation

For the implementation, the apache spark tools are used with scala, since Apache Spark is a unified analysis engine for the processing of big data, what we are looking for in the course, with integrated modules for transmission, SQL, machine learning and graphics processing.

By having the aforementioned characteristics, it allows us to carry out everything proposed in this document. In order to have a good analysis of the aforementioned learning algorithms and make a comparison between them.

You will work with the scala programming language because scala combines functional and object-oriented programming in a concise high-level language.

Scala's static types help avoid errors in complex applications, and its JVM and JavaScript runtimes allow you to build high-performance systems with easy access to huge library ecosystems.

As mentioned above, scala helps JVM (Java Virtual Machine) runtimes, this is useful to us since we will make the comparison between different learning algorithms and scala helps us to execute them in a better way.

The implementation is done on a desktop computer with a Linux base operating system, in the source code editor "Visual Code".

### Code to implement

- #### Support Vector Machine

  ```scala
  // 1. Import the "LinearSVC" library, this binary classifier optimizes the hinge loss using the OWLQN optimizer. 
  import org.apache.spark.ml.classification.LinearSVC
  import org.apache.spark.ml.feature.VectorAssembler
  import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
  import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

  // 2. Import session.
  import org.apache.spark.sql.SparkSession
  val spark = SparkSession.builder.getOrCreate()

  // 3. Load the training data. 
  val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/Files/bank-full.csv")
    
  // 4. Process of categorizing the variables type string to numeric. 
  val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
  val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
  val newcolumn = no.withColumn("y",'y.cast("Int"))

  // 5. Vector is created with the column features 
  val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

  // 7. Transforms into a new df 
  val data2 = assembler.transform(newcolumn)

  // 8. Column and label are given a new name 
  val featuresLabel = data2.withColumnRenamed("y", "label")

  // 9. Select index
  val dataIndexed = featuresLabel.select("label","features")
  // Index columns

  // 10. Split the data into training and test sets (30% held out for testing).
  val Array(training, test) = dataIndexed.randomSplit(Array(0.7, 0.3))

  // 11. Set the maximum number of iterations and the regularization parameter .
  val lsvc = new LinearSVC().setMaxIter(10).setRegParam(0.1)

  // 12. Make a fit to adjust the model.
  val supportVM = new LinearSVC().setMaxIter(10).setRegParam(0.1)
  val model = supportVM.fit(training)
  val predictions = model.transform(test)
  val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
  val metrics = new MulticlassMetrics(predictionAndLabels)

  // 13. Print LSVC
  println("Confusion matrix:")
  println(metrics.confusionMatrix)
  println("Accuracy: " + metrics.accuracy) 
  println(s"Test Error = ${(1.0 - metrics.accuracy)}")

  ```
- #### Decison Tree
  ```scala
  // 1. Import libraries
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.classification.DecisionTreeClassificationModel
    import org.apache.spark.ml.classification.DecisionTreeClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler}

    // 2. Import session
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder.appName("DecisionTreeClassificationExample").getOrCreate()

    // 3. Load the data stored in LIBSVM format as a DataFrame.
    val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/Files/bank-full.csv")

    // 4. Process of categorizing the variables type string to numeric.
    val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
    val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
    val newcolumn = no.withColumn("y",'y.cast("Int"))

    // 5. Vector is created with the column features
    val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

    // 6. Transforms into a new df
    val data2 = assembler.transform(newcolumn)

    // 7. Column and label are given a new name
    val featuresLabel = data2.withColumnRenamed("y", "label")

    // 8. Select index
    val dataIndexed = featuresLabel.select("label","features")
    // Index columns

    // 9. Index labels, adding metadata to the label column.
    //    Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel").fit(dataIndexed)

    // 10. Automatically identify categorical features, and index them.
    val featureIndexer = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(4).fit(dataIndexed)
    // features with > 4 distinct values are treated as continuous.

    // 11. Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

    // 12. Train a DecisionTree model.
    val dt = new DecisionTreeClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures")
    // 13. Convert indexed labels back to original labels.
    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(labelIndexer.labels)

    // 14. Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline().setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

    // 15. Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // 16. Make predictions.
    val predictions = model.transform(testData)

    // 17. Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(5)

    // 18. Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
  

  ```

- #### Logistic Regression

  ```scala
  // Import libraries
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
    import org.apache.spark.ml.linalg.Vectors

    // 2. Import session
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder.getOrCreate()

    // 3. Load the data stored in LIBSVM format as a DataFrame.
    val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/Files/bank-full.csv")

    // 4. Process of categorizing the variables type string to numeric.
    val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
    val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
    val newcolumn = no.withColumn("y",'y.cast("Int"))

    // 5. Vector is created with the column features
    val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

    // 7. Transforms into a new df
    val data2 = assembler.transform(newcolumn)

    // 8. Column and label are given a new name
    val featuresLabel = data2.withColumnRenamed("y", "label")

    // 9. Select index
    val dataIndexed = featuresLabel.select("label","features")
    // Index columns

    // 10. Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = dataIndexed.randomSplit(Array(0.7, 0.3))

    // 11. Logistic Regression
    val logisticReg = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8).setFamily("multinomial")
    val model = logisticReg.fit(trainingData)
    val predictions = model.transform(testData)
    val predictionAndLabels = predictions.select($"prediction",$"label").as[(Double, Double)].rdd
    val metrics = new MulticlassMetrics(predictionAndLabels)

    // 12. Print results
    println("Confusion matrix:")
    println(metrics.confusionMatrix)
    println("Accuracy: " + metrics.accuracy)
    println(s"Test Error: ${(1.0 - metrics.accuracy)}")

  ```

- #### Multilayer Perceptron


  ```scala
  
   // 1. Import libraries
    import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.linalg.Vectors
    import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer, VectorAssembler}

    // 2. Import session
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

    // 3. Load the data stored in LIBSVM format as a DataFrame.
    val data  = spark.read.option("header","true").option("inferSchema", "true").option("delimiter",";").format("csv").load("/Files/bank-full.csv")

    // 4. Process of categorizing the variables type string to numeric.
    val yes = data.withColumn("y",when(col("y").equalTo("yes"),1).otherwise(col("y")))
    val no = yes.withColumn("y",when(col("y").equalTo("no"),2).otherwise(col("y")))
    val newcolumn = no.withColumn("y",'y.cast("Int"))

    // 5. Vector is created with the column features
    val assembler = new VectorAssembler().setInputCols(Array("balance","day","duration","pdays","previous")).setOutputCol("features")

    // 6. Transforms into a new df
    val data2 = assembler.transform(newcolumn)

    // 7. Column and label are given a new name
    val featuresLabel = data2.withColumnRenamed("y", "label")

    // 8. Select index
    val dataIndexed = featuresLabel.select("label","features")
    // Index columns

    // 9. Split the data into train and test
    val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // 10. Specify layers for the neural network:
    //    input layer of size 5 (features), two intermediate of size 2 and 2
    //    and output of size 4 (classes)
    val layers = Array[Int](5,2,2,4)

    // 11. Create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

    // 12. Train the model
    val model = trainer.fit(train)

    // 13. Compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
    println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
  ```

## Results
### Accuracy
Accuracy expressed in percent.
<html><div align="center"><img src="https://i.ibb.co/b7KP8rs/Screenshot-2021-06-23-at-16-11-34-Libro1-pdf-1.png"></div></html>

### Execution time
Execution time expressed in seconds. 
<html><div align="center"><img src="https://i.ibb.co/5jTqCDM/Screenshot-2021-06-23-at-16-13-21-Libro2-pdf-1.png"></div></html>

### Observations of the results

As we can see after obtaining the results we see that in terms of average Decision Tree obtains the most favorable results speaking of precision, although not by much difference but in turn it is in the penultimate place in runtime, with Support vector machine being the one with the least time it takes you to perform the calculations.

Each algorithm varies its execution time within 30 executions but never leaving very specific ranges, although each algorithm is created for specific situations, this can give us an example of how efficient they are in various situations.

## Conclusions
When using the different algorithms, we could see that although they all have a somewhat different workflow, they can show very similar results, the difference in this case varies in seconds and less than 1% in precision in almost the general averages.

I think this work gives us an idea of which algorithm works best for certain situations thinking on a larger scale (a greater number of data), for this situation the best was Decision Tree which leans more for precision over execution time, In the real world, there may be cases where the time in which the results are obtained is above precision, considering that only a minimal percentage is lost.

