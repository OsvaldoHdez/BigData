## Exam Unit 2
 Se importan las librerías de MLlib necesarias para realizar la clasificación con Multilayer Perceptron
```scala
import org.apache.spark.sql.SparkSession
val spar = SparkSession.builder()

import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.linalg.Vectors


```

Aqui se cargan los datos del dataset iris.csv en la variable "data"
```scala

val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load("Iris.csv")

// Se eliminan los campos null
val dataClean = data.na.drop()

```

Punto 2 se muestran nombres de las columnas
```scala
dataClean.columns
scala> dataClean.columns
res0: Array[String] = Array(sepal_length, sepal_width, petal_length, petal_width, species)
```

Punto 3 se imprime el esquema de los datos
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

 Punto 4 imprime las primeras 5 columnas
```scala
dataClean.show(5)
dataClean.select($"sepal_length",$"sepal_width",$"petal_length",$"petal_width",$"species").show(5)

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

Punto 5 metodo describe
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
Punto 6 transformación pertinente para los datos categoricos los cuales seran nuestras etiquetas a clasificar.
```scala
// Se declara un vector que transformara los datos a la variable "features"
val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))

scala> val vectorFeatures = (new VectorAssembler().setInputCols(Array("sepal_length","sepal_width", "petal_length","petal_width")).setOutputCol("features"))
vectorFeatures: org.apache.spark.ml.feature.VectorAssembler = vecAssembler_82088f126176

// Se transforman los features usando el dataframe
val features = vectorFeatures.transform(dataClean)

scala> val features = vectorFeatures.transform(dataClean)
features: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 4 more fields]

// Se declara un "StringIndexer" que transformada los datos en "species" en datos numericos
val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")

scala> val speciesIndexer = new StringIndexer().setInputCol("species").setOutputCol("label")
speciesIndexer: org.apache.spark.ml.feature.StringIndexer = strIdx_f953afe94d4a

// Ajustamos las especies indexadas con el vector features
val dataIndexed = speciesIndexer.fit(features).transform(features)

scala> val dataIndexed = speciesIndexer.fit(features).transform(features)
dataIndexed: org.apache.spark.sql.DataFrame = [sepal_length: double, sepal_width: double ... 5 more fields]
```
 Punto 7  Construya el modelos de clasificación y explique su arquitectura.
```scala
// Con la variable "splits" hacemos un corte de forma aleatoria
val splits = dataIndexed.randomSplit(Array(0.6, 0.4), seed = 1234L)

// Se declara la variable "train" la cual tendra el 60% de los datos
val train = splits(0)

// Se declara la variable "test" la cual tendra el 40% de los datos
val test = splits(1)

// Se establece la configuración de las capas para el modelo de redes neuronales artificiales
// De entrada: 4 (features)
// las dos Intermedias  5 y 4 respectivamente
// de salida 3, al tener 3 tipos de clases
val layers = Array[Int](4, 5, 4, 3)

// Se configura el entrenador del algoritmo Multilayer con sus respectivos parametros
val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

// Se entrena el modelo con los datos de entrenamiento
val model = trainer.fit(train)

// Se prueban ya entrenado el modelo
val result = model.transform(test)


```

 Punto 8 Imprima los resultados del modelo, Se selecciona la predicción y la etiqueta que seran guardado en la variable
```scala
val predictionAndLabels = result.select("prediction", "label")

// Ee muestran algunos datos de la predicción contra los reales para ver resultados
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

Se ejecuta la estimacion de la precision del modelo

```scala
val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
println(s"Prueba de precision = ${evaluator.evaluate(predictionAndLabels)}")

scala> println(s"Prueba de precision = ${evaluator.evaluate(predictionAndLabels)}")
Prueba de precision = 0.9607843137254902

```


