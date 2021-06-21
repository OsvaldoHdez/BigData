# Exam U2
### 0. Import libraries and spark session
- #### Code
```scala
// Import libraries
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.Pipeline

// Import session
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder.appName("MultilayerPerceptronClassifier").getOrCreate()
```

### 1. Load iris dataframe and data clean
- #### Code
```scala
// Load iris dataframe
val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("iris.csv")

// Null fields are removed
val dataClean = data.na.drop()
```

### 2. Show columns name
- #### Code and results
```scala
dataClean.columns
    scala> dataClean.columns
    res0: Array[String] = Array(sepal_length, sepal_width, petal_length, petal_width, species)
```

### 3. Print data schema
- #### Code and results
```scala
dataClean.printSchema()

    scala> dataClean.printSchema()
    root
    |-- sepal_length: double (nullable = true)
    |-- sepal_width: double (nullable = true)
    |-- petal_length: double (nullable = true)
    |-- petal_width: double (nullable = true)
    |-- species: string (nullable = true)
```

### 4. Print the first five columns 
- #### Code and results
```scala
dataClean.show(5) // 1
dataClean.select($"sepal_length",$"sepal_width",$"petal_length",$"petal_width",$"species").show(5) // 2

    scala> dataClean.show(5)
    +------------+-----------+------------+-----------+-------+
    |sepal_length|sepal_width|petal_length|petal_width|species|
    +------------+-----------+------------+-----------+-------+
    |         5.1|        3.5|         1.4|        0.2| setosa|
    |         4.9|        3.0|         1.4|        0.2| setosa|
    |         4.7|        3.2|         1.3|        0.2| setosa|
    |         4.6|        3.1|         1.5|        0.2| setosa|
    |         5.0|        3.6|         1.4|        0.2| setosa|
    +------------+-----------+------------+-----------+-------+
    only showing top 5 rows
```

### 5. Use the describe () method to learn more about the data in the DataFrame. 
- #### Code and results
```scala
dataClean.describe().show()

    scala> dataClean.describe().show()
    +-------+------------------+-------------------+------------------+------------------+---------+
    |summary|      sepal_length|        sepal_width|      petal_length|       petal_width|  species|
    +-------+------------------+-------------------+------------------+------------------+---------+
    |  count|               150|                150|               150|               150|      150|
    |   mean| 5.843333333333335| 3.0540000000000007|3.7586666666666693|1.1986666666666672|     null|
    | stddev|0.8280661279778637|0.43359431136217375| 1.764420419952262|0.7631607417008414|     null|
    |    min|               4.3|                2.0|               1.0|               0.1|   setosa|
    |    max|               7.9|                4.4|               6.9|               2.5|virginica|
    +-------+------------------+-------------------+------------------+------------------+---------+
```

### 6. Transformation for the categorical data which will be our labels to be classified.
- #### Code and results
```scala
// A vector is declared that transforms the data to the variable "features" 
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

    scala> val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).   setOutputCol("features"))
    vectorFeatures: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_82088f126176

// Features are transformed using the dataframe 
val features = vectorFeatures.transform(dataClean)

    scala> val features = vectorFeatures.transform(dataClean)
    features: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]

// A "StringIndexer" is declared that transforms the data in "species" into numeric data 
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

    scala> val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
    speciesIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_f953afe94d4a

// Adjust the indexed species with the vector features 
val dataIndexed = speciesIndexer.fit(features).transform(features)

    scala> val dataIndexed = speciesIndexer.fit(features).transform(features)
    dataIndexed: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 5 more fields]
```

### 7. Build the classification model.
- #### Code and results
```scala
// With the variable "splits" we make a random cut 
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

// The variable "train" is declared which will have 60% of the data 
val train = splits(0)

// The variable "test" is declared which will have 40% of the data 
val test = splits(1)

// Layer settings are established for the artificial neural network model 
// Input: 4 (features) 
// the two Intermediates 5 and 4 respectively, output 3, having 3 types of classes 
val layers = Array[Int](4, 5, 4, 3)

// The Multilayer algorithm trainer is configured with its respective parameters 
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// The model is trained with the training data 
val model = trainer.fit(train)

// The model is tested already trained 
val result = model.transform(test)
```

### 8. Print the model results.
- #### Code and results
```scala
// The prediction and the label that will be saved in the variable are selected 
val predictionAndLabels = result.select("prediction", "label")

// Some data of the prediction is shown against the real ones to see results 
predictionAndLabels.show(50)

    scala> predictionAndLabels.show(50)
    +----------+-----+
    |prediction|label|
    +----------+-----+
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       0.0|  0.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       0.0|  0.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       2.0|  2.0|
    |       0.0|  0.0|
    |       2.0|  2.0|
    |       0.0|  0.0|
    |       1.0|  1.0|
    |       0.0|  0.0|
    |       0.0|  0.0|
    |       1.0|  1.0|
    |       2.0|  2.0|
    |       1.0|  1.0|
    |       0.0|  0.0|
    |       0.0|  0.0|
    |       0.0|  0.0|
    |       0.0|  0.0|
    |       0.0|  1.0|
    |       0.0|  0.0|
    |       1.0|  1.0|
    |       1.0|  1.0|
    |       0.0|  0.0|
    |       1.0|  1.0|
    |       0.0|  1.0|
    |       0.0|  0.0|
    |       1.0|  1.0|
    |       1.0|  1.0|
    |       0.0|  0.0|
    |       1.0|  1.0|
    |       1.0|  1.0|
    |       1.0|  1.0|
    |       1.0|  1.0|
    |       1.0|  1.0|
    |       0.0|  0.0|
    |       1.0|  1.0|
    |       0.0|  0.0|
    +----------+-----+
    only showing top 50 rows


result.show(30)

    scala> result.show(30)
    +------------+-----------+------------+-----------+----------+-----------------+-----+--------------------+--------------------+----------+
    |sepal_length|sepal_width|petal_length|petal_width|   species|         features|label|       rawPrediction|         probability|prediction|
    +------------+-----------+------------+-----------+----------+-----------------+-----+--------------------+--------------------+----------+
    |         4.3|        3.0|         1.1|        0.1|    setosa|[4.3,3.0,1.1,0.1]|  2.0|[15.6939102472457...|[4.49726715588115...|       2.0|
    |         4.4|        2.9|         1.4|        0.2|    setosa|[4.4,2.9,1.4,0.2]|  2.0|[15.7205776664764...|[5.32055408135416...|       2.0|
    |         4.4|        3.0|         1.3|        0.2|    setosa|[4.4,3.0,1.3,0.2]|  2.0|[15.7038134381871...|[4.78697261626649...|       2.0|
    |         4.6|        3.4|         1.4|        0.3|    setosa|[4.6,3.4,1.4,0.3]|  2.0|[15.6877056144553...|[4.32476070392186...|       2.0|
    |         4.6|        3.6|         1.0|        0.2|    setosa|[4.6,3.6,1.0,0.2]|  2.0|[15.6739436876185...|[3.96538767140727...|       2.0|
    |         4.7|        3.2|         1.6|        0.2|    setosa|[4.7,3.2,1.6,0.2]|  2.0|[15.7412734317472...|[6.06200464499190...|       2.0|
    |         4.8|        3.1|         1.6|        0.2|    setosa|[4.8,3.1,1.6,0.2]|  2.0|[15.7753958949252...|[7.51682313582786...|       2.0|
    |         5.0|        3.2|         1.2|        0.2|    setosa|[5.0,3.2,1.2,0.2]|  2.0|[15.7415843992936...|[6.07389961264308...|       2.0|
    |         5.0|        3.3|         1.4|        0.2|    setosa|[5.0,3.3,1.4,0.2]|  2.0|[15.7498146125576...|[6.39734401981583...|       2.0|
    |         5.0|        3.4|         1.5|        0.2|    setosa|[5.0,3.4,1.5,0.2]|  2.0|[15.7461427849689...|[6.25096742565720...|       2.0|
    |         5.0|        3.6|         1.4|        0.2|    setosa|[5.0,3.6,1.4,0.2]|  2.0|[15.7112740691320...|[5.01748555668546...|       2.0|
    |         5.1|        2.5|         3.0|        1.1|versicolor|[5.1,2.5,3.0,1.1]|  0.0|[34.5866091698007...|[1.0,2.5941767559...|       0.0|
    |         5.1|        3.4|         1.5|        0.2|    setosa|[5.1,3.4,1.5,0.2]|  2.0|[15.7634953365452...|[6.97354848103035...|       2.0|
    |         5.1|        3.8|         1.5|        0.3|    setosa|[5.1,3.8,1.5,0.3]|  2.0|[15.6983940392728...|[4.62619671582550...|       2.0|
    |         5.2|        2.7|         3.9|        1.4|versicolor|[5.2,2.7,3.9,1.4]|  0.0|[37.1529115232574...|[0.99999999999999...|       0.0|
    |         5.2|        4.1|         1.5|        0.1|    setosa|[5.2,4.1,1.5,0.1]|  2.0|[15.7019884862697...|[4.73221773470795...|       2.0|
    |         5.3|        3.7|         1.5|        0.2|    setosa|[5.3,3.7,1.5,0.2]|  2.0|[15.7470383515724...|[6.28635720081642...|       2.0|
    |         5.4|        3.4|         1.5|        0.4|    setosa|[5.4,3.4,1.5,0.4]|  2.0|[15.7786365063372...|[7.67195856461062...|       2.0|
    |         5.5|        2.3|         4.0|        1.3|versicolor|[5.5,2.3,4.0,1.3]|  0.0|[37.1529131667946...|[0.99999999999999...|       0.0|
    |         5.5|        4.2|         1.4|        0.2|    setosa|[5.5,4.2,1.4,0.2]|  2.0|[15.7014941439117...|[4.71749385627412...|       2.0|
    |         5.6|        2.9|         3.6|        1.3|versicolor|[5.6,2.9,3.6,1.3]|  0.0|[37.1384680193605...|[0.99999999999999...|       0.0|
    |         5.7|        2.5|         5.0|        2.0| virginica|[5.7,2.5,5.0,2.0]|  1.0|[45.6464753798674...|[9.46112527590297...|       1.0|
    |         5.7|        2.9|         4.2|        1.3|versicolor|[5.7,2.9,4.2,1.3]|  0.0|[37.1528363365649...|[0.99999999999999...|       0.0|
    |         5.8|        2.7|         3.9|        1.2|versicolor|[5.8,2.7,3.9,1.2]|  0.0|[37.1495583430124...|[0.99999999999999...|       0.0|
    |         5.8|        2.8|         5.1|        2.4| virginica|[5.8,2.8,5.1,2.4]|  1.0|[45.6471650570526...|[9.40997336577198...|       1.0|
    |         5.8|        4.0|         1.2|        0.2|    setosa|[5.8,4.0,1.2,0.2]|  2.0|[15.7424946718691...|[6.10885321133428...|       2.0|
    |         5.9|        3.0|         5.1|        1.8| virginica|[5.9,3.0,5.1,1.8]|  1.0|[45.2902823047448...|[1.55536651002909...|       1.0|
    |         6.0|        2.2|         4.0|        1.0|versicolor|[6.0,2.2,4.0,1.0]|  0.0|[37.1481683596418...|[0.99999999999999...|       0.0|
    |         6.0|        2.7|         5.1|        1.6|versicolor|[6.0,2.7,5.1,1.6]|  0.0|[41.5263470563781...|[0.52317501968582...|       0.0|
    |         6.0|        2.9|         4.5|        1.5|versicolor|[6.0,2.9,4.5,1.5]|  0.0|[37.1529849481000...|[0.99999999999999...|       0.0|
    +------------+-----------+------------+-----------+----------+-----------------+-----+--------------------+--------------------+----------+
```

##### The estimation of the precision of the model is executed 
```scala
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Prueba de precision = ${evaluator.evaluate(predictionAndLabels)}")

    scala> println(s"Prueba de precision = ${evaluator.evaluate(predictionAndLabels)}")
    Prueba de precision = 0.9607843137254902
```
